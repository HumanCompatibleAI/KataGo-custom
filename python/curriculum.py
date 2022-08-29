#!/usr/bin/python3

"""Curriculum module for using in victimplay."""

import argparse
import datetime
import enum
import json
import logging
import os
import pathlib
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from sgfmill import sgf

Config = Mapping[str, Any]


@enum.unique
class Color(enum.Enum):
    """Color of Go stones (black or white)."""

    BLACK = "B"
    WHITE = "W"

    @staticmethod
    def from_string(color: str) -> "Color":
        color = color.upper()
        return Color(color)


def flip_color(color: Color) -> Color:
    if color == Color.BLACK:
        return Color.WHITE
    elif color == Color.WHITE:
        return Color.BLACK
    else:
        raise TypeError("Color must be black or white, not {color}")


@dataclass(frozen=True)
class AdvGameInfo:
    """Class for storing game result from the adversary perspective."""

    victim_name: str
    victim_visits: int
    game_hash: str
    winner: Optional[bool]
    score_diff: float
    score_wo_komi_diff: float


@dataclass(frozen=True)
class PlayerStat:
    """Class for storing game statistics.

    Statistics is being represented from the adversary perspective.
    """

    name: Optional[str] = None
    win_rate: Optional[float] = None
    score_diff: Optional[float] = None
    score_wo_komi_diff: Optional[float] = None
    policy_loss: Optional[float] = None

    def get_stat_members(self) -> Dict[str, float]:
        d = asdict(self)
        del d["name"]
        return d


@dataclass(frozen=True)
class VictimCriteria(PlayerStat):
    """Criteria for the victim change.

    Victim is represented by the model name and max visits.
    Criteria are represented by the statistics members.
    For victim only one criterion can be enabled, all others should be None.
    """

    max_visits_victim: Optional[int] = None
    max_visits_adv: Optional[int] = None

    def get_stat_members(self) -> Dict[str, float]:
        d = super().get_stat_members()
        del d["max_visits_victim"]
        del d["max_visits_adv"]
        return d

    def __post_init__(self):
        if self.name is None:
            raise ValueError("VictimCriteria: victim name is None")
        criteria = self.get_stat_members()
        enabled_criteria = [v for k, v in criteria.items() if v is not None]
        num_enabled = len(enabled_criteria)
        if num_enabled != 1:
            msg = f"Need 1 criteria enabled, got {num_enabled}: {enabled_criteria}"
            raise ValueError(msg)

    # check if adv_stat has a greater value of enabled criteria
    def check_if_gt(self, adv_stat: PlayerStat) -> bool:
        criteria = self.get_stat_members()
        adv_vals = adv_stat.get_stat_members()
        for k, v in criteria.items():
            if v is not None:
                logging.info(
                    "{}: {} (adv) <-> {} (threshold)".format(k, adv_vals[k], v),
                )
                if adv_vals[k] > v:
                    return True
        return False

    def matches_criteria(
        self,
        other: Union["VictimCriteria", Mapping[str, Any]],
    ) -> bool:
        """Check current victim against the latest parameters.

        Args:
            other: A VictimCriteria, or mapping with the keys
                "name", "max_visits_victim", "max_visits_adv".

        Returns:
            True if the above fields compare equal; False otherwise.

        Parameters can be either VictimCriteria or a dict.
        """
        d0 = asdict(self)
        if isinstance(other, VictimCriteria):
            d1 = asdict(other)
        else:
            d1 = other

        fields = ("name", "max_visits_victim", "max_visits_adv")
        return all(d0[f] == d1[f] for f in fields)


def is_name_victim(name: str) -> bool:
    """Has 'victim-' in `name` and not a colored evaluator."""
    return "victim-" in name and "__" not in name


def get_game_hash(game: sgf.Sgf_game) -> Optional[str]:
    """Extracts the hash from the game comment field C."""
    # game_c = startTurnIdx=N,initTurnNum=N,gameHash=X
    try:
        game_c = game.get_root().get("C")
    except KeyError:
        logging.warning("No comment field in game %s", game)
        return None
    # game_hash_raw: gameHash=X
    game_hash_raw = game_c.split(",")[2]
    # Returns the hash X
    return game_hash_raw.split("=")[1]


def get_game_score(game: sgf.Sgf_game) -> Optional[float]:
    try:
        result = game.get_root().get("RE")
    except KeyError:
        logging.warning("No result (RE tag) present in SGF game: '%s'", game)
        return None
    try:
        win_score = result.split("+")[1]
    except IndexError:
        logging.warning("No winner in result '%s'", result)
        return None
    try:
        win_score = float(win_score)
    except ValueError:
        logging.warning("Game score is not numeric: '%s'", win_score)
        return None
    return win_score


def get_victim_adv_colors(game: sgf.Sgf_game) -> Tuple[str, int, Color, Color]:
    """Returns a tuple of victim name, visit count, and victim and adversary color."""
    colors: Sequence[Color] = (Color.BLACK, Color.WHITE)
    name_to_colors: Mapping[str, Color] = {
        game.get_player_name(color.value.lower()): color for color in colors
    }
    victim_names = [name for name in name_to_colors.keys() if is_name_victim(name)]
    if len(victim_names) != 1:
        raise ValueError("Found '{len(victim_names)}' != 1 victims: %s", victim_names)
    victim_name = victim_names[0]
    assert victim_name.startswith("victim-")

    victim_color = name_to_colors[victim_name]
    adv_color = flip_color(victim_color)

    victim_name = victim_name[7:]
    visit_key = victim_color.value + "R"  # BR or WR: black/white rank
    game_root = game.get_root()
    victim_visits = int(game_root.get(visit_key).lstrip("v"))

    return victim_name, victim_visits, victim_color, adv_color


def get_game_info(sgf_str: str) -> Optional[AdvGameInfo]:
    try:
        game = sgf.Sgf_game.from_string(sgf_str)
    except IndexError:
        logging.warning("Error parsing game: '%s'", sgf_str)
        return None

    game_hash = get_game_hash(game)
    win_score = get_game_score(game)
    if game_hash is None or win_score is None:
        return None

    victim_name, victim_visits, victim_color, adv_color = get_victim_adv_colors(game)
    win_color = Color.from_string(game.get_winner())

    komi = game.get_komi()
    adv_komi = komi if adv_color == Color.WHITE else -komi

    if win_color is None:  # tie (should never happen under default rules)
        adv_minus_victim_score = 0
        adv_minus_victim_score_wo_komi = 0
        winner = None
    else:
        winner = win_color == adv_color
        adv_minus_victim_score = win_score if winner else -win_score
        adv_minus_victim_score_wo_komi = adv_minus_victim_score - adv_komi

    return AdvGameInfo(
        victim_name=victim_name,
        victim_visits=victim_visits,
        game_hash=game_hash,
        winner=winner,
        score_diff=adv_minus_victim_score,
        score_wo_komi_diff=adv_minus_victim_score_wo_komi,
    )


def get_files_sorted_by_modification_time(
    folder: pathlib.Path,
    extension: Optional[str] = None,
    ignore_extensions: Optional[Sequence[str]] = None,
) -> Sequence[str]:
    all_files = []
    for path, dirnames, filenames in os.walk(folder, followlinks=True):
        for f in filenames:
            ext = os.path.splitext(f)[1]
            if ignore_extensions is not None and ext in ignore_extensions:
                continue
            if extension is None or ext == extension:
                file_path = os.path.join(path, f)
                all_files.append([file_path, os.path.getmtime(file_path)])
    # sort from newest to oldest
    all_files.sort(key=lambda x: x[1], reverse=True)

    # leave file names only
    return [x[0] for x in all_files]


def recompute_statistics(
    games: List[AdvGameInfo],
    games_for_compute: int,
    current_victim_name: str,
    current_victim_visits: Optional[int],
) -> Optional[PlayerStat]:
    """Compute statistics from games played by the current victim."""
    # don't have enough data
    if len(games) < games_for_compute:
        logging.info(f"Incomplete statistics, got only {len(games)} games")
        return None
    logging.info("Computing {} games".format(len(games)))

    games_cur_victim = []
    for game in games:
        same_victim = game.victim_name == current_victim_name
        same_visits = current_victim_visits is None
        same_visits = same_visits or game.victim_visits == current_victim_visits
        if same_victim and same_visits:
            games_cur_victim.append(game)

    if len(games_cur_victim) < len(games):
        logging.info(
            "Incomplete statistics for current victim, got only {} games".format(
                len(games_cur_victim),
            ),
        )
        return None

    sum_wins = 0
    sum_ties = 0
    sum_score = 0
    sum_score_wo_komi = 0
    for game in games:
        # game.winner can be None (for ties), but a tie is still not a win
        if game.winner:
            sum_wins += 1
        elif game.winner is None:
            sum_ties += 1
        sum_score += game.score_diff
        sum_score_wo_komi += game.score_wo_komi_diff

    logging.info(
        "Got {} wins and {} ties from {} games".format(sum_wins, sum_ties, len(games)),
    )
    win_rate = float(sum_wins) / len(games)
    mean_diff_score = float(sum_score) / len(games)
    mean_diff_score_wo_komi = float(sum_score_wo_komi) / len(games)

    return PlayerStat(
        name=current_victim_name,
        win_rate=win_rate,
        score_diff=mean_diff_score,
        score_wo_komi_diff=mean_diff_score_wo_komi,
    )


class Curriculum:
    """Curriculum object.

    Curriculum is used for updating victims for victimplay based on
    the criteria specified in the provided config.
    """

    def __init__(
        self,
        victims_input_dir: pathlib.Path,
        victims_output_dir: pathlib.Path,
        config: Optional[Sequence[Config]] = None,
        config_json: Optional[str] = None,
        config_json_file: Optional[pathlib.Path] = None,
    ):
        """Initial curriculum setup.

        Construct and initialize curriculum.

        Args:
            victims_input_dir: The folder with all victim model
                files specified in the config.
            victims_output_dir: The folder where we copy victims for selfplay.
            config: List of victims.
            config_json: Serialized JSON list of victims.
            config_json_file: JSON file with list of victims.

        Raises:
            ValueError: Empty configuration.
        """
        self.MAX_VICTIM_COPYING_EFFORTS = 10
        self.VICTIM_COPY_FILESYSTEM_ACCESS_TIMEOUT = 10
        self.SELFPLAY_CONFIG_OVERRIDE_NAME = "victim.cfg"

        self.stat_files = []
        self.sgf_games = []
        self.game_hashes = dict()

        if config_json_file is not None:
            logging.info(f"Curriculum: loading JSON config from '{config_json_file}'")
            with open(config_json_file) as f:
                config = json.load(f)
        elif config_json is not None:
            logging.info("Curriculum: loading JSON config from a string")
            config = json.loads(config_json)
        elif config is not None:
            logging.info("Using python list as a config")

        if not config:
            raise ValueError("Empty config for the curriculum play!")

        self.victims_input_dir = victims_input_dir
        self.victims_output_dir = victims_output_dir
        self.selfplay_config_override_path = (
            victims_output_dir / self.SELFPLAY_CONFIG_OVERRIDE_NAME
        )
        self.victims_output_dir_tmp = victims_output_dir.with_name(
            victims_output_dir.name + "_tmp",
        )

        self.victim_idx = 0
        self.finished = False
        self.victims: List[VictimCriteria] = []
        for line in config:
            line = dict(line)
            line.pop("_comment", None)  # delete _comment if it exists
            try:
                cond = VictimCriteria(**line)
            except ValueError as e:
                raise ValueError(f"Invalid victim config '{line}'") from e
            self.victims.append(cond)

        logging.info("Loaded curriculum with the following params:")
        logging.info("\n".join([str(x) for x in config]))

        logging.info("Finding the latest victim...")
        victim_files = get_files_sorted_by_modification_time(
            self.victims_output_dir,
            ignore_extensions=(".cfg", ".conf"),
        )
        if victim_files:
            last_victim_name = os.path.basename(victim_files[0])
            victim_params = {
                "name": last_victim_name,
                "max_visits_victim": None,
                "max_visits_adv": None,
            }

            # find current maxVisits settings
            if os.path.exists(self.selfplay_config_override_path):
                with open(self.selfplay_config_override_path) as f:
                    for line in f.readlines():
                        line = line.strip()
                        if not line:
                            continue
                        name, val = line.split("=")
                        if name == "maxVisits0":
                            victim_params["max_visits_victim"] = int(val)
                        elif name == "maxVisits1":
                            victim_params["max_visits_adv"] = int(val)

            # determine current victim-with-max-visits index
            victim_found = False
            for cur_idx in range(len(self.victims)):
                if self.victims[cur_idx].matches_criteria(victim_params):
                    self.victim_idx = cur_idx
                    victim_found = True
                    break

            if not victim_found:
                logging.warning(
                    "Victim '{}' is not found in '{}', starting from scratch".format(
                        str(victim_params),
                        self.victims_output_dir,
                    ),
                )

        logging.info(
            "Copying the latest victim '{}'...".format(self._cur_victim),
        )
        self._try_victim_copy()
        logging.info("Curriculum initial setup is complete")

    @property
    def _cur_victim(self) -> VictimCriteria:
        return self.victims[self.victim_idx]

    def _update_victim_config(self):
        tmp_path = self.victims_output_dir_tmp / self.SELFPLAY_CONFIG_OVERRIDE_NAME
        with open(tmp_path, "w") as f:
            if self._cur_victim.max_visits_victim is not None:
                f.write(f"maxVisits0={self._cur_victim.max_visits_victim}\n")
            if self._cur_victim.max_visits_adv is not None:
                f.write(f"maxVisits1={self._cur_victim.max_visits_adv}\n")
        shutil.move(str(tmp_path), self.selfplay_config_override_path)

    def _try_victim_copy(self, force_if_exists=False):
        victim_name = self._cur_victim.name
        victim_path = self.victims_output_dir / victim_name
        victim_path_tmp = self.victims_output_dir_tmp / victim_name

        if not force_if_exists and os.path.exists(victim_path):
            return

        # Attempt to copy
        for _ in range(self.MAX_VICTIM_COPYING_EFFORTS):
            try:
                # Make sure directories exist
                os.makedirs(self.victims_output_dir, exist_ok=True)
                os.makedirs(self.victims_output_dir_tmp, exist_ok=True)
                self._update_victim_config()

                # We copy to a tmp directory then move to make the overall
                # operation atomic, which is needed to avoid race conditions
                # with the C++ code.
                shutil.copy(
                    self.victims_input_dir / victim_name,
                    victim_path_tmp,
                )
                shutil.move(str(victim_path_tmp), victim_path)
                return
            except OSError:
                logging.warning(
                    "Cannot copy victim '{}', maybe "
                    "filesystem problem? Waiting {} sec...".format(
                        self._cur_victim.name,
                        self.VICTIM_COPY_FILESYSTEM_ACCESS_TIMEOUT,
                    ),
                )
                time.sleep(self.VICTIM_COPY_FILESYSTEM_ACCESS_TIMEOUT)

        raise RuntimeError(
            "Problem copying victim '{}', curriculum stopped".format(
                self._cur_victim.name,
            ),
        )

    def try_move_on(
        self,
        adv_stat: PlayerStat,
        policy_loss: Optional[float] = None,
    ):
        if self.finished:
            return

        logging.info("Checking whether we need to move to the next victim...")
        want_victim_update = False
        if self._cur_victim.check_if_gt(adv_stat):
            want_victim_update = True
        if policy_loss is not None:
            raise NotImplementedError("Policy loss check is not implemented yet")

        if not want_victim_update:
            return

        self.victim_idx += 1
        if self.victim_idx == len(self.victims):
            self.finished = True
            return

        logging.info("Moving to the next victim '{}'".format(self._cur_victim.name))
        self._try_victim_copy(True)

    def update_sgf_games(self, selfplay_dir: pathlib.Path, games_for_compute: int):
        all_sgfs = get_files_sorted_by_modification_time(selfplay_dir, ".sgfs")

        useful_files = set()
        cur_games = []
        for sgf_file in all_sgfs:
            if sgf_file not in self.game_hashes:
                self.game_hashes[sgf_file] = set()

            with open(sgf_file) as f:
                logging.debug("Processing SGF file '{}'".format(sgf_file))
                all_lines = list(f.readlines())

                for line in reversed(all_lines):
                    if not line.endswith("\n"):  # game not fully written
                        continue
                    sgf_string = line.strip()
                    game_stat = get_game_info(sgf_string)
                    if game_stat is None:
                        continue

                    # game hash was found, so consider that the rest of them are older
                    # so stop scanning this file
                    if game_stat.game_hash in self.game_hashes[sgf_file]:
                        break

                    self.game_hashes[sgf_file].add(game_stat.game_hash)
                    cur_games.append(game_stat)
                    useful_files.add(sgf_file)

        # now have cur_games sorted from newer to older
        logging.info(
            "Got {} new games from {} files".format(len(cur_games), len(useful_files)),
        )
        for f in useful_files:
            logging.info("Useful SGF file: '{}'".format(str(f)))

        # insert new games in the beginning
        self.sgf_games[:0] = cur_games

        # leave only games_for_compute games for statistics computation
        # so delete some old games
        if len(self.sgf_games) > games_for_compute:
            del self.sgf_games[games_for_compute:]

    """
    Run curriculum checking.
    @param selfplay_dir: Folder with selfplay results.
    @param games_for_compute: Number of games to compute statistics.
    @param checking_periodicity: Checking interval in seconds.
    """

    def checking_loop(
        self,
        selfplay_dir: pathlib.Path,
        games_for_compute: int,
        checking_periodicity: int,
    ):
        logging.info("Starting curriculum loop")
        while True:
            self.update_sgf_games(selfplay_dir, games_for_compute)
            adv_stat = recompute_statistics(
                self.sgf_games,
                games_for_compute,
                self._cur_victim.name,
                self._cur_victim.max_visits_victim,
            )
            if adv_stat is not None:
                self.try_move_on(adv_stat=adv_stat)
                if self.finished:
                    logging.info("Curriculum is done. Stopping")
                    break
            logging.info(
                "Curriculum is alive, current victim : {} @ v{}".format(
                    self._cur_victim.name,
                    self._cur_victim.max_visits_victim,
                ),
            )
            time.sleep(checking_periodicity)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run victim replacement based on win rate.",
    )
    parser.add_argument(
        "-selfplay-dir",
        type=pathlib.Path,
        required=True,
        help="Directory with selfplay data",
    )
    parser.add_argument(
        "-input-models-dir",
        type=pathlib.Path,
        required=True,
        help="Input dir with victim model files",
    )
    parser.add_argument(
        "-output-models-dir",
        type=pathlib.Path,
        required=True,
        help="Output dir for adding new victims",
    )
    parser.add_argument(
        "-games-for-compute",
        type=int,
        required=False,
        default=1000,
        help="Number of last games for statistics computation",
    )
    parser.add_argument(
        "-checking-periodicity",
        type=int,
        required=False,
        default=60,
        help="Statistics computation periodicity in seconds",
    )
    parser.add_argument(
        "-config-json-string",
        required=False,
        help="Curriculum JSON config with " "victims sequence (JSON content)",
    )
    parser.add_argument(
        "-config-json-file",
        type=pathlib.Path,
        default="configs/curriculum_conf.json",
        help="Curriculum JSON config with " "victims sequence (JSON file path)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Set log level to DEBUG (default INFO)",
        action="store_const",
        dest="log_level",
        const=logging.DEBUG,
        default=logging.INFO,
    )

    return parser.parse_args()


def setup_logging(log_level: int) -> None:
    """Setup logging to file /outputs/curriculum-<timestamp>.log and stdout."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    timestamp = datetime.datetime.utcnow().isoformat()
    file_handler = logging.FileHandler(filename=f"/outputs/curriculum-{timestamp}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stdout_handler)


def make_curriculum(args: argparse.Namespace) -> Curriculum:
    """Construct curriculum from CLI `args`."""
    if args.config_json_file is not None:
        return Curriculum(
            args.input_models_dir,
            args.output_models_dir,
            config_json_file=args.config_json_file,
        )
    elif args.config_json_string is not None:
        return Curriculum(
            args.input_models_dir,
            args.output_models_dir,
            config_json=args.config_json_string,
        )
    else:
        raise ValueError(
            "Curriculum: either path to JSON config or "
            "JSON config string must be provided",
        )


def main():
    """Main console entry point to script."""
    args = parse_args()
    setup_logging(args.log_level)
    curriculum = make_curriculum(args)

    try:
        curriculum.checking_loop(
            args.selfplay_dir,
            args.games_for_compute,
            args.checking_periodicity,
        )
    # we really want to silence 'B902: blind except'
    # because we want a stacktrace and error description in logs
    except BaseException as e:  # noqa: B902
        logging.exception("Curriculum error: {}".format(e))
        raise

    logging.info("Curriculum finished!")


if __name__ == "__main__":
    main()
