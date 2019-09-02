def load_questions(file_path, line_parser):
    with open(file_path, 'rb') as fp:
        line = fp.readline()
        qs = []
        while line:
            question = line_parser(line)
            if question:
                qs.append(question)
            line = fp.readline()
    return qs

def parser(line):
    temp = line.decode("utf-8").strip().split(";")
    question = temp[0]
    answer = temp[1]
    r1, e1, r2, r3, e2 = question.split("\t")
    answers = answer.split("\t")[1:]
    return (answers, r1, e1, r2, r3, e2)

questions = load_questions("countries_test.txt", parser)
with open("test.txt", "w+") as f:
    content = ""
    for question in questions:
        answers, r1, e1, r2, r3, e2 = question
        line = e2 + "\t" + r3 + "\t" + answers[0] + "\n"
        content += line
    f.write(content)