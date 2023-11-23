import os


def test_python_cmd_with_blank():
    path1 = "data/1 Folder/"
    cmd = """ls \"{0}\"""".format(path1)
    print("cmd", cmd)
    os.system(cmd)

    cmd = 'ls "{0}"'.format(path1)
    print("cmd", cmd)
    os.system(cmd)


test_python_cmd_with_blank()
