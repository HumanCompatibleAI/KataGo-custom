#ifndef SEARCH_SEARCHPARAMS_H_
#define SEARCH_SEARCHPARAMS_H_

#include "../core/global.h"
#include "../game/board.h"

struct SearchParams {
  // Modifications of pass suppression behavior.
  enum class PassingBehavior {
    // Essentially use vanilla MCTS to determine when passing makes sense
    Standard,
    // Pass when the only legal alternatives are to play in your own pass-alive territory
    AvoidPassAliveTerritory,
    // Pass when the only legal alternatives are in territory your opponent "almost certainly" (95% chance) owns,
    // or that are "much worse" than passing
    LastResort,
    // Disallow passing when this would cause you to lose on the next turn by Tromp-Taylor scoring if the opponent
    // passes. Instead of trusting what the neural net says we use "oracle" access to the TT score.
    NoSuicide,
    // Passing is only allowed when the net thinks it has a safe win margin
    OnlyWhenAhead,
    // Passing is only allowed when the net is likely losing
    OnlyWhenBehind
  };
  static PassingBehavior strToPassingBehavior(const std::string& behaviorStr);
  static std::string passingBehaviorToStr(PassingBehavior behavior);
  PassingBehavior passingBehavior;
  // If enabled, then we will definitely pass if it wins us the game.
  bool forceWinningPass;

  // Algorithm to use for search
  enum class SearchAlgorithm {
    // Vanilla self-play MCTS
    MCTS,
    // A-MCTS-S: Adversarial MCTS with sampling
    AMCTS_S,
    // A-MCTS-S++: Adversarial MCTS with sampling and averaging over board symmetries
    AMCTS_SXX,
    // A-MCTS-R: Adversarial MCTS with recursion
    AMCTS_R
  };
  static SearchAlgorithm strToSearchAlgo(const std::string& algoStr);
  static std::string searchAlgoToStr(SearchAlgorithm algo);
  SearchAlgorithm searchAlgo;
  std::string getSearchAlgoAsStr() const;
  bool usingAdversarialAlgo() const;

  // Overrides the number of visits we use in AMCTS-R to simulate the victim.
  std::optional<int> oppVisitsOverride;

  // If non-none, determines whether to set the weight of opponent nodes to zero.
  // By default, this will be false for MCTS and true for adversarial algorithms.
  std::optional<bool> oppWeightZeroingOverride;

  //Utility function parameters
  double winLossUtilityFactor;     //Scaling for [-1,1] value for winning/losing
  double staticScoreUtilityFactor; //Scaling for a [-1,1] "scoreValue" for having more/fewer points, centered at 0.
  double dynamicScoreUtilityFactor; //Scaling for a [-1,1] "scoreValue" for having more/fewer points, centered at recent estimated expected score.
  double dynamicScoreCenterZeroWeight; //Adjust dynamic score center this proportion of the way towards zero, capped at a reasonable amount.
  double dynamicScoreCenterScale; //Adjust dynamic score scale. 1.0 indicates that score is cared about roughly up to board sizeish.
  double noResultUtilityForWhite; //Utility of having a no-result game (simple ko rules or nonterminating territory encore)
  double noResultUtility; //Utility of having a no-result game, regardless of player's color.
  double drawEquivalentWinsForWhite; //Consider a draw to be this many wins and one minus this many losses.

  // Typically no-result is only allowed under certain rule sets, and the
  // no-result logit is cleared out under incompatible rule sets.
  // If "hitTurnLimitIsNoResult" is enabled then this may no longer be true.
  // Setting this param means that no-result logits are no longer cleared.
  bool forceAllowNoResultPredictions;

  //Search tree exploration parameters
  double cpuctExploration;  //Constant factor on exploration, should also scale up linearly with magnitude of utility
  double cpuctExplorationLog; //Constant factor on log-scaling exploration, should also scale up linearly with magnitude of utility
  double cpuctExplorationBase; //Scale of number of visits at which log behavior starts having an effect

  double cpuctUtilityStdevPrior;
  double cpuctUtilityStdevPriorWeight;
  double cpuctUtilityStdevScale;

  double fpuReductionMax;   //Max amount to reduce fpu value for unexplore children
  double fpuLossProp; //Scale fpu this proportion of the way towards assuming a move is a loss.

  bool fpuParentWeightByVisitedPolicy; //For fpu, blend between parent average and parent nn value based on proportion of policy visited.
  double fpuParentWeightByVisitedPolicyPow; //If fpuParentWeightByVisitedPolicy, what power to raise the proportion of policy visited for blending.
  double fpuParentWeight; //For fpu, 0 = use parent average, 1 = use parent nn value, interpolates between.

  double policyOptimism; //Interpolate geometrically between raw policy and optimistic policy

  //Tree value aggregation parameters
  double valueWeightExponent; //Amount to apply a downweighting of children with very bad values relative to good ones
  bool useNoisePruning; //For computation of value, prune out weight that greatly exceeds what is justified by policy prior
  double noisePruneUtilityScale; //The scale of the utility difference at which useNoisePruning has effect
  double noisePruningCap; //Maximum amount of weight that noisePruning can remove

  //Uncertainty weighting
  bool useUncertainty; //Weight visits by uncertainty
  double uncertaintyCoeff; //The amount of visits weight that an uncertainty of 1 utility is.
  double uncertaintyExponent; //Visits weight scales inversely with this power of the uncertainty
  double uncertaintyMaxWeight; //Add minimum uncertainty so that the most weight a node can have is this

  //Graph search
  bool useGraphSearch; //Enable graph search instead of tree search?
  int graphSearchRepBound; //Rep bound to use for graph search transposition safety. Higher will reduce transpositions but be more safe.
  double graphSearchCatchUpLeakProb; //Chance to perform a visit to deepen a branch anyways despite being behind on visit count.
  //double graphSearchCatchUpProp; //When sufficiently far behind on visits on a transposition, catch up extra by adding up to this fraction of parents visits at once.

  //Root parameters
  bool rootNoiseEnabled;
  double rootDirichletNoiseTotalConcentration; //Same as alpha * board size, to match alphazero this might be 0.03 * 361, total number of balls in the urn
  double rootDirichletNoiseWeight; //Policy at root is this weight * noise + (1 - this weight) * nn policy

  double rootPolicyTemperature; //At the root node, scale policy probs by this power
  double rootPolicyTemperatureEarly; //At the root node, scale policy probs by this power, early in the game
  double rootFpuReductionMax; //Same as fpuReductionMax, but at root
  double rootFpuLossProp; //Same as fpuLossProp, but at root
  int rootNumSymmetriesToSample; //For the root node, sample this many random symmetries (WITHOUT replacement) and average the results together.
  bool rootSymmetryPruning; //For the root node, search only one copy of each symmetrically equivalent move.
  //We use the min of these two together, and also excess visits get pruned if the value turns out bad.
  double rootDesiredPerChildVisitsCoeff; //Funnel sqrt(this * policy prob * total visits) down any given child that receives any visits at all at the root

  double rootPolicyOptimism; //Interpolate geometrically between raw policy and optimistic policy

  //Parameters for choosing the move to play
  double chosenMoveTemperature; //Make move roughly proportional to visit count ** (1/chosenMoveTemperature)
  double chosenMoveTemperatureEarly; //Temperature at start of game
  double chosenMoveTemperatureHalflife; //Halflife of decay from early temperature to temperature for the rest of the game, scales for board sizes other than 19.
  double chosenMoveSubtract; //Try to subtract this many visits from every move prior to applying temperature
  double chosenMovePrune; //Outright prune moves that have fewer than this many visits

  bool useLcbForSelection; //Using LCB for move selection?
  bool useLcbForSelfplayMove; //Use LCB to make moves during self-play?
  double lcbStdevs; //How many stdevs a move needs to be better than another for LCB selection
  double minVisitPropForLCB; //Only use LCB override when a move has this proportion of visits as the top move
  bool useNonBuggyLcb; //LCB was very minorly buggy as of pre-v1.8. Set to true to fix.

  //Mild behavior hackery
  double rootEndingBonusPoints; //Extra bonus (or penalty) to encourage good passing behavior at the end of the game.
  bool rootPruneUselessMoves; //Prune moves that are entirely useless moves that prolong the game.
  bool conservativePass; //Never assume one's own pass will end the game.
  bool fillDameBeforePass; //When territory scoring, heuristically discourage passing before filling the dame.
  Player avoidMYTDaggerHackPla; //Hacky hack to avoid a particular pattern that gives some KG nets some trouble. Should become unnecessary in the future.
  double wideRootNoise; //Explore at the root more widely
  bool enablePassingHacks; //Enable some hacks that mitigate rare instances when passing messes up deeper searches.

  double playoutDoublingAdvantage; //Play as if we have this many doublings of playouts vs the opponent
  Player playoutDoublingAdvantagePla; //Negate playoutDoublingAdvantage when making a move for the opponent of this player. If empty, opponent of the root player.

  double avoidRepeatedPatternUtility; //Have the root player avoid repeating similar shapes, penalizing this much utility per instance.

  float nnPolicyTemperature; //Scale neural net policy probabilities by this temperature, applies everywhere in the tree
  bool antiMirror; //Enable anti-mirroring logic

  double subtreeValueBiasFactor; //Dynamically adjust neural net utilties based on empirical stats about their errors in search
  int32_t subtreeValueBiasTableNumShards; //Number of shards for subtreeValueBiasFactor for initial hash lookup and mutexing
  double subtreeValueBiasFreeProp; //When a node is no longer part of the relevant search tree, only decay this proportion of the weight.
  double subtreeValueBiasWeightExponent; //When computing empiricial bias, weight subtree results by childvisits to this power.

  //Threading-related
  int nodeTableShardsPowerOfTwo; //Controls number of shards of node table for graph search transposition lookup
  double numVirtualLossesPerThread; //Number of virtual losses for one thread to add

  //Asyncbot
  int numThreads; //Number of threads
  double minPlayoutsPerThread; //If the number of playouts to perform per thread is smaller than this, cap the number of threads used.
  int64_t maxVisits; //Max number of playouts from the root to think for, counting earlier playouts from tree reuse
  int64_t maxPlayouts; //Max number of playouts from the root to think for, not counting earlier playouts from tree reuse
  double maxTime; //Max number of seconds to think for

  //Same caps but when pondering
  int64_t maxVisitsPondering;
  int64_t maxPlayoutsPondering;
  double maxTimePondering;

  //Amount of time to reserve for lag when using a time control
  double lagBuffer;

  //Human-friendliness
  double searchFactorAfterOnePass; //Multiply playouts and visits and time by this much after a pass by the opponent
  double searchFactorAfterTwoPass; //Multiply playouts and visits and time by this after two passes by the opponent

  //Time control
  double treeReuseCarryOverTimeFactor; //Assume we gain this much "time" on the next move purely from % tree preserved * time spend on that tree.
  double overallocateTimeFactor; //Prefer to think this factor longer than recommended by base level time control
  double midgameTimeFactor; //Think this factor longer in the midgame, proportional to midgame weight
  double midgameTurnPeakTime; //The turn considered to have midgame weight 1.0, rising up from 0.0 in the opening, for 19x19
  double endgameTurnTimeDecay; //The scale of exponential decay of midgame weight back to 1.0, for 19x19
  double obviousMovesTimeFactor; //Think up to this factor longer on obvious moves, weighted by obviousness
  double obviousMovesPolicyEntropyTolerance; //What entropy does the policy need to be at most to be (1/e) obvious?
  double obviousMovesPolicySurpriseTolerance; //What logits of surprise does the search result need to be at most to be (1/e) obvious?

  double futileVisitsThreshold; //If a move would not be able to match this proportion of the max visits move in the time or visit or playout cap remaining, prune it.


  SearchParams();
  ~SearchParams();

  void printParams(std::ostream& out);

  //Params to use for testing, with some more recent values representative of more real use (as of Jan 2019)
  static SearchParams forTestsV1();
  //Params to use for testing, with some more recent values representative of more real use (as of Mar 2022)
  static SearchParams forTestsV2();

  static void failIfParamsDifferOnUnchangeableParameter(const SearchParams& initial, const SearchParams& dynamic);
};

#endif  // SEARCH_SEARCHPARAMS_H_
