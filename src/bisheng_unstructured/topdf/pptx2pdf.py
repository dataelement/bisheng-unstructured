import os
import shutil
import signal
import subprocess


class PptxToPDF(object):
    def __init__(self, kwargs={}):
        cmd_template = """
            soffice --headless --convert-to pdf --outdir \"{1}\" \"{0}\"
        """

        def _norm_cmd(cmd):
            return " ".join([p.strip() for p in cmd.strip().split()])

        self.cmd_template = _norm_cmd(cmd_template)

    @staticmethod
    def run(cmd):
        try:
            p = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            raise Exception("timeout in transforming pptx to pdf")
        except Exception as e:
            raise Exception(f"err in pptx2pdf: [{e}]")

    def render(self, input_file, output_file=None, to_bytes=False):
        type_ext = input_file.rsplit(".", 1)[-1]
        filename = os.path.basename(input_file)
        temp_dir = os.path.dirname(input_file)
        output_filename = filename.rsplit(".", 1)[0] + ".pdf"
        temp_output_file = os.path.join(temp_dir, output_filename)

        assert type_ext in ["pptx", "ppt"]

        cmd = self.cmd_template.format(input_file, temp_dir)
        PptxToPDF.run(cmd)

        if output_file is not None:
            shutil.move(temp_output_file, output_file)

        if to_bytes:
            return open(temp_output_file, "rb").read()
