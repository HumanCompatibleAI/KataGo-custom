import re
import json

b_regex = re.compile("B\[[a-z]{0,2}\]")
w_regex = re.compile("W\[[a-z]{0,2}\]")

SGF_TO_GTP_COL = {
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
    "e": "E",
    "f": "F",
    "g": "G",
    "h": "H",
    "i": "J",
    "j": "K",
    "k": "L",
    "l": "M",
    "m": "N",
    "n": "O",
    "o": "P",
    "p": "Q",
    "q": "R",
    "r": "S",
    "s": "T",
}
r
SGF_TO_GTP_ROW = {
    "a": "1",
    "b": "2",
    "c": "3",
    "d": "4",
    "e": "5",
    "f": "6",
    "g": "7",
    "h": "8",
    "i": "9",
    "j": "10",
    "k": "11",
    "l": "12",
    "m": "13",
    "n": "14",
    "o": "15",
    "p": "16",
    "q": "17",
    "r": "18",
    "s": "19",
}

target_file = "block_test_position.sgf"
board_size = 19

info_last_chance = {}
with open("info_last_chance.jsonl", "r") as f:
    for line in f:
        info_last_chance.update(json.loads(line))

target_move = info_last_chance[target_file]["move_number"]

with open(target_file, "r") as f:
    contents = f.read()

sgf_moves_list = contents.split(";")
sgf_moves_list = sgf_moves_list[2:]  # remove the game info

gtp_movelist = []
for i, move in enumerate(sgf_moves_list):
    try:
        if i % 2 == 0:  # black"s turn
            sgf_move = re.findall(b_regex, move)[0]
            if len(sgf_move) == 5:
                gtp_move = [
                    "B",
                    SGF_TO_GTP_COL[sgf_move[2]]
                    + str(board_size + 1 - int(SGF_TO_GTP_ROW[sgf_move[3]])),
                ]
            elif len(sgf_move) == 3:
                gtp_move = ["B", "pass"]

        else:  # white"s turn
            sgf_move = re.findall(w_regex, move)[0]
            if len(sgf_move) == 5:
                gtp_move = [
                    "W",
                    SGF_TO_GTP_COL[sgf_move[2]]
                    + str(board_size + 1 - int(SGF_TO_GTP_ROW[sgf_move[3]])),
                ]
            elif len(sgf_move) == 3:
                gtp_move = ["W", "pass"]

        gtp_movelist.append(gtp_move)
    except:
        print(i % 2)
        print(move)
        print(sgf_move)
        exit()

analysis_query = {
    "id": "test_query",
    "moves": gtp_movelist,
    "rules": "tromp-taylor",
    "komi": 7.5,
    "boardXSize": board_size,
    "boardYSize": board_size,
    "analyzeTurns": [target_move],
    "maxVisits": 5000,
}

# print(json.dumps(analysis_query))

with open(f'input_{target_file.split(".")[0]}.txt', "w") as f:
    print(json.dumps(analysis_query), file=f)