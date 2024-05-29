#include "../neuralnet/pytorchbackend.h"

#include <cassert>
#include <sstream>

#include <ATen/autocast_mode.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>

#include "../core/fileutils.h"
#include "../core/global.h"
#include "../neuralnet/modelversion.h"

using namespace torch::indexing;

namespace TorchNeuralNet {

namespace {

// HACK(tomtseng): We should write the model version and max model board size
// when exporting the model. For now we'll just hard code the values.
constexpr int MAX_BOARD_LEN = 19;
constexpr int MODEL_VERSION = 14;
const int NUM_SPATIAL_FEATURES = NNModelVersion::getNumSpatialFeatures(MODEL_VERSION);
const int NUM_GLOBAL_FEATURES = NNModelVersion::getNumGlobalFeatures(MODEL_VERSION);

void logModelForwardFailure(ComputeHandle* handle, InputBuffers* inputBuffers) {
  if (handle->logger != nullptr) {
    std::stringstream str;
    str << "Model evaluation failed with model " << getModelName(&handle->model) << " on input:";
    for (const auto& input: inputBuffers->modelInputs) {
      str << '\n' << input;
    }
    handle->logger->write(str.str());
  }
}

}  // namespace


LoadedModel::LoadedModel(const std::string& filename)
  : model(torch::jit::load(filename))
  , modelName(filename) {}

LoadedModel::LoadedModel(const LoadedModel& other)
  : model(other.model.clone())
  , modelName(other.modelName) {}

LoadedModel* loadModelFile(const std::string& file, const std::string& expectedSha256) {
  if (expectedSha256.size() != 0) {
    throw StringError("Checking sha256 for PyTorch models is not yet implemented.\n");
  }
  if (!FileUtils::exists(file)) {
    throw IOError("File does not exist: " + file);
  }
  return new LoadedModel(file);
}

void freeLoadedModel(LoadedModel* model) {
  delete model;
}

std::string getModelName(const LoadedModel* model) {
  return model->modelName;
}

int getModelVersion(const LoadedModel*) {
  return MODEL_VERSION;
}

ComputeContext::ComputeContext(int nnXLen_, int nnYLen_, bool useFP16_)
  : nnXLen(nnXLen_)
  , nnYLen(nnYLen_)
  , useFP16(useFP16_) {
}

ComputeContext* createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const std::string& openCLTunerFile,
  const std::string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;
  if (useNHWCMode != enabled_t::False) {
    throw StringError("useNHWC is not yet implemented for PyTorch.");
  }
  const bool useFP16 = useFP16Mode == enabled_t::True;
  assert(nnXLen <= MAX_BOARD_LEN);
  assert(nnYLen <= MAX_BOARD_LEN);

  ComputeContext* context = new ComputeContext(nnXLen, nnYLen, useFP16);
  return context;
}

void freeComputeContext(ComputeContext* context) {
  delete context;
}

ComputeHandle::ComputeHandle(
    const ComputeContext* context,
    const LoadedModel* model_,
    Logger* logger_,
    int maxBatchSize_,
    int gpuIdxForThisThread
)
  : model(*model_)
  , device(torch::Device(at::kCUDA, gpuIdxForThisThread))
  , logger(logger_)
  , maxBatchSize(maxBatchSize_)
  , nnXLen(context->nnXLen)
  , nnYLen(context->nnYLen)
  , useFP16(context->useFP16) {
    model.model.eval();
    model.model.to(device);
  }

ComputeHandle* createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  (void)requireExactNNLen;
  (void)serverThreadIdx;

  if (inputsUseNHWC) {
    throw StringError("inputsUseNHWC is not yet implemented for PyTorch.");
  }
  if (gpuIdxForThisThread == -1) {
    gpuIdxForThisThread = 0;
  }

  return new ComputeHandle(context, loadedModel, logger, maxBatchSize, gpuIdxForThisThread);
}

void freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

InputBuffers::InputBuffers(int maxBatchSize)
  : hostSpatialInputs(torch::empty({maxBatchSize, NUM_SPATIAL_FEATURES, MAX_BOARD_LEN, MAX_BOARD_LEN}))
  , hostGlobalInputs(torch::empty({maxBatchSize, NUM_GLOBAL_FEATURES})) {
  const size_t NUM_INPUTS = 2;
  modelInputs.reserve(NUM_INPUTS);
}

InputBuffers* createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  (void)loadedModel;
  (void)nnXLen;
  (void)nnYLen;
  return new InputBuffers(maxBatchSize);
}

void freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

void getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  std::vector<NNOutput*>& outputs
) {
  const int batchSize = numBatchEltsFilled;
  assert(batchSize <= gpuHandle->maxBatchSize);
  assert(batchSize > 0);
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  if (nnXLen != MAX_BOARD_LEN || nnYLen != MAX_BOARD_LEN) {
    // The PyTorch model assumes that smaller board sizes' inputs are formatted
    // like in the following example channel-0 spatial input (signifying which
    // locations are on the board) for a 5x5 input:
    //   1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   ...
    //   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //
    // If nnXLen and nnYLen are set to 5 instead of MAX_BOARD_LEN==19,
    // KataGo populates the inputs as
    //   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    //   1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   ...
    //   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //
    // The other backends handle this but I (tomtseng) haven't investigated how.
    // For now we'll just enforce that nnXLen and nnYLen are 19. If a user wants
    // to play on a 5x5 board, they should include 19 in the bSizes config
    // parameter (and set its bSizeRelProbs to 0 if they don't actually want any
    // 19x19 games), otherwise we throw an exception here.
    throw StringError(Global::strprintf("Board len not yet supported: %d x %d", nnXLen, nnYLen));
  }
  constexpr bool INPUTS_USE_NHWC = false;

  const auto& spatialInputs = inputBuffers->hostSpatialInputs;
  const auto& globalInputs = inputBuffers->hostGlobalInputs;
  for (int row = 0; row < batchSize; row++) {
    SymmetryHelpers::copyInputsWithSymmetry(inputBufs[row]->rowSpatial, spatialInputs[row].data_ptr<float>(), 1, nnYLen, nnXLen, NUM_SPATIAL_FEATURES, INPUTS_USE_NHWC, inputBufs[row]->symmetry);
    const float* rowGlobal = inputBufs[row]->rowGlobal;
    std::copy(rowGlobal, rowGlobal + NUM_GLOBAL_FEATURES, globalInputs[row].data_ptr<float>());
  }

  const auto modelDataType = gpuHandle->useFP16 ? torch::kFloat16 : torch::kFloat32;
  auto& modelInputs = inputBuffers->modelInputs;
  modelInputs.clear();
  modelInputs.emplace_back(spatialInputs.index({Slice(0, batchSize)}).to(gpuHandle->device, modelDataType));
  modelInputs.emplace_back(globalInputs.index({Slice(0, batchSize)}).to(gpuHandle->device, modelDataType));

  c10::IValue modelOutput;
  {
    torch::NoGradGuard no_grad;
    try {
      modelOutput = gpuHandle->model.model.forward(modelInputs);
    } catch (const c10::Error&) {
      logModelForwardFailure(gpuHandle, inputBuffers);
      throw;
    } catch (const std::runtime_error& err) {
      if (std::string(err.what()).find("RuntimeError: Input type") != std::string::npos) {
        // If the error message looks like "RuntimeError: Input type
        // (CUDAHalfType) and weight type (CUDAFloatType) should be the same",
        // the error may be that the user did not set useFP16-N to match whether
        // TorchScript model nnModelFileN was exported with FP16.
        if (gpuHandle->logger != nullptr) {
          gpuHandle->logger->write("HINT: Is useFP16 set correctly for each TorchScript bot?");
        }
      } else {
        logModelForwardFailure(gpuHandle, inputBuffers);
      }
      throw;
    }
  }
  const auto& modelOutputs = modelOutput.toTupleRef().elements();
  at::Tensor policyOutputs = modelOutputs[0].toTensor();
  const at::Tensor& valueOutputs = modelOutputs[1].toTensor().to(at::kCPU);
  const at::Tensor& miscValueOutputs = modelOutputs[2].toTensor().to(at::kCPU);
  const at::Tensor& moreMiscValueOutputs = modelOutputs[3].toTensor().to(at::kCPU);
  at::Tensor ownershipOutputs;

  const bool has_optimistic_policy = policyOutputs.size(1) > 1;
  at::Tensor policies = policyOutputs.index({Slice(), 0});
  at::Tensor optimisticPolicyDiffs;
  if (has_optimistic_policy) {
    optimisticPolicyDiffs = policyOutputs.index({Slice(), 1}).sub_(policies);
  }
  for (int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];

    const int numPolicyValues = nnYLen * nnXLen + 1;
    at::Tensor policy = policies.index({row, Slice(0, numPolicyValues)});
    if (has_optimistic_policy) {
      const float policyOptimism = (float)inputBufs[row]->policyOptimism;
      // final policy = policy + (policy - optimisticPolicy) * policyOptimism
      policy.add_(optimisticPolicyDiffs.index({row, Slice(0, numPolicyValues)}), policyOptimism);
    }
    policy = policy.to(at::kCPU, torch::kFloat32).contiguous();
    SymmetryHelpers::copyOutputsWithSymmetry(policy.data_ptr<float>(), output->policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    // Copy the policy output for passing as well.
    output->policyProbs[nnYLen * nnXLen] = policy[nnYLen * nnXLen].item<float>();

    const auto& valueOutput = valueOutputs[row];
    output->whiteWinProb = valueOutput[0].item<float>();
    output->whiteLossProb = valueOutput[1].item<float>();
    output->whiteNoResultProb = valueOutput[2].item<float>();

    const auto& miscValueOutput = miscValueOutputs[row];
    output->whiteScoreMean = miscValueOutput[0].item<float>();
    output->whiteScoreMeanSq = miscValueOutput[1].item<float>();
    output->whiteLead = miscValueOutput[2].item<float>();
    output->varTimeLeft = miscValueOutput[3].item<float>();

    const auto& moreMiscValueOutput = moreMiscValueOutputs[row];
    output->shorttermWinlossError = moreMiscValueOutput[0].item<float>();
    output->shorttermScoreError = moreMiscValueOutput[1].item<float>();

    if (output->whiteOwnerMap != NULL) {
      if (!ownershipOutputs.defined()) {
        ownershipOutputs = modelOutputs[4].toTensor().to(at::kCPU, torch::kFloat32);
      }
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipOutputs[row].data_ptr<float>(), output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }
  }
}

}  // namespace TorchNeuralNet
