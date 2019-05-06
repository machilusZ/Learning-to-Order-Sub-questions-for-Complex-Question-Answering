import json

with open("wc_graph.txt") as fp:
    line = fp.readline()
    es = []
    rs = []
    label = {}
    while line:
        e1, r, e2 = line[0:-1].split("\t")
        if e1 not in label:
            if r == "is_in_country":
                label[e1] = "team"
            if r == "is_in_country_inverse":
                label[e1] = "country"
            if r == "plays_position_inverse":
                label[e1] = "position"
            if r == "plays_in_club_inverse":
                label[e1] = "club"
            if r == "plays_position":
                label[e1] = "player"
            if r == "wears_number":
                label[e1] = "player"
            if r == "plays_for_country":
                label[e1] = "player"
            if r == "plays_in_club":
                label[e1] = "player"
            if r == "plays_for_country_inverse":
                label[e1] = "country"
            if r == "is_aged":
                label[e1] = "player"
        if e2 not in label:
            if r == "is_in_country":
                label[e2] = "country"
            if r == "is_in_country_inverse":
                label[e2] = "team"
            if r == "plays_position_inverse":
                label[e2] = "player"
            if r == "plays_in_club_inverse":
                label[e2] = "player"
            if r == "plays_position":
                label[e2] = "position"
            if r == "wears_number":
                label[e2] = "number"
            if r == "plays_for_country":
                label[e2] = "country"
            if r == "plays_in_club":
                label[e2] = "club"
            if r == "plays_for_country_inverse":
                label[e2] = "player"
            if r == "is_aged":
                label[e2] = "age"
        line = fp.readline()
with open("label.json", "w") as outfile:
        json.dump(label, outfile)