import os
import shutil
import signal
import subprocess
import tempfile

import openpyxl


class ExcelToPDF(object):
    def __init__(self, kwargs={}):
        cmd_template = """
          soffice -env:SingleAppInstance=\"false\" -env:UserInstallation=\"file://{1}\" --convert-to html --outdir \"{1}\" \"{0}\"
        """
        cmd_template2 = """
            soffice --headless -env:SingleAppInstance=\"false\" -env:UserInstallation=\"file://{1}\" --convert-to xlsx --outdir \"{1}\" \"{0}\"
        """

        cmd_template3 = 'sed -e \'s/\t/,/g\' "{0}" > "{1}"'

        cmd_template4 = """
                wkhtmltopdf --disable-javascript --disable-local-file-access --disable-external-links --no-images "{0}" "{1}"
                """

        def _norm_cmd(cmd):
            return " ".join([p.strip() for p in cmd.strip().split()])

        self.cmd_template = _norm_cmd(cmd_template)
        self.cmd_template2 = _norm_cmd(cmd_template2)
        self.cmd_template3 = cmd_template3
        self.cmd_template4 = cmd_template4

    @staticmethod
    def run(cmd):
        try:
            p = subprocess.Popen(
                cmd,
                shell=True,
                preexec_fn=os.setsid,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            exit_code = p.wait(timeout=30)
            if exit_code != 0:
                stdout, stderr = p.communicate()
                raise Exception(
                    f"err in excel2pdf: return code is {exit_code}, stderr: {stderr}, stdout: {stdout}"
                )
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            raise Exception(f"timeout in transforming xlsx to pdf, cmd: [{cmd}]")
        except Exception as e:
            raise Exception(f"err in excel2pdf: [{e}]")

    def render(self, input_file, output_file=None, to_bytes=False):
        type_ext = input_file.rsplit(".", 1)[-1]
        filename = os.path.basename(input_file)
        output_filename = filename.rsplit(".", 1)[0] + ".pdf"
        assert type_ext in ["xlsx", "xls", "tsv", "csv"]

        with tempfile.TemporaryDirectory() as temp_dir:
            if type_ext in ["tsv"]:
                csv_filename = filename.rsplit(".", 1)[0] + ".csv"
                output_csv = os.path.join(temp_dir, csv_filename)
                cmd = self.cmd_template3.format(input_file, output_csv)
                ExcelToPDF.run(cmd)
                input_file = output_csv
                type_ext = "csv"

            if type_ext in ["xls", "csv"]:
                cmd = self.cmd_template2.format(input_file, temp_dir)
                ExcelToPDF.run(cmd)
                filename = filename.rsplit(".", 1)[0] + ".xlsx"
                input_file = os.path.join(temp_dir, filename)

            temp_output_file = os.path.join(temp_dir, output_filename)
            wb = openpyxl.load_workbook(input_file)
            for ws in wb:
                ws.print_options.gridLines = True
                ws.print_options.gridLinesSet = True

            input_file = os.path.join(temp_dir, filename)
            wb.save(input_file)

            # 先把excel转为html
            cmd = self.cmd_template.format(input_file, temp_dir)
            ExcelToPDF.run(cmd)
            html_file_path = os.path.join(temp_dir, filename.rsplit(".", 1)[0] + ".html")
            with open(html_file_path, "r+", encoding="utf-8") as f:
                html_content = f.readlines()
                for index, one in enumerate(html_content):
                    if one.find("text/css") != -1:
                        html_content.insert(
                            index + 1,
                            "table {word-break: break-word;}\ntable td{word-break: break-all;border: 1px solid #000000; padding: 3px;}",
                        )
                        break
                f.seek(0)
                f.writelines(html_content)

            # 在把html转成pdf
            cmd = self.cmd_template4.format(html_file_path, temp_output_file)
            ExcelToPDF.run(cmd)

            if output_file is not None:
                shutil.move(temp_output_file, output_file)

            if to_bytes:
                return open(temp_output_file, "rb").read()
