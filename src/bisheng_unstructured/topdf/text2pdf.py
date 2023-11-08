import os


class Text2PDF(object):
    def __init__(self, kwargs={}):
        cmd_template = """
          pandoc -o {1} --pdf-engine=xelatex {0}
              -V mainfont="Alibaba PuHuiTi"
              -V sansfont="Alibaba PuHuiTi"
              -V monofont="Cascadia Mono"
              -V CJKmainfont="Alibaba PuHuiTi"
              -V CJKsansfont="Alibaba PuHuiTi"
              -V CJKmonofont="Adobe Heiti Std"
        """

        self.cmd_template2 = "pandoc -o {1} {0}"

        def _norm_cmd(cmd):
            return " ".join([p.strip() for p in cmd.strip().split()])

        self.cmd_template = _norm_cmd(cmd_template)
        self.cmd_template2 = "pandoc -o {1} {0}"

    @staticmethod
    def run(cmd):
        try:
            exit_code = os.system(cmd)
            if exit_code != 0:
                raise Exception("error in transforming xlsx to pdf")
        except Exception as e:
            raise e

    def render(self, input_file, output_file=None, to_bytes=False):
        type_ext = input_file.rsplit(".", 1)[-1]
        filename = os.path.basename(input_file)
        temp_dir = os.path.dirname(input_file)

        assert type_ext in ["txt", "md", "html"]

        if type_ext == "txt":
            cmd = self.cmd_template2.format(input_file, "./data/xxxx.md")
            Text2PDF.run(cmd)
            input_file = "./data/xxxx.md"

        if output_file is None:
            output_filename = filename.rsplit(".", 1)[0] + ".pdf"
            output_file = os.path.join(temp_dir, output_filename)

        cmd = self.cmd_template.format(input_file, output_file)
        Text2PDF.run(cmd)

        if to_bytes:
            return open(output_file, "rb").read()
