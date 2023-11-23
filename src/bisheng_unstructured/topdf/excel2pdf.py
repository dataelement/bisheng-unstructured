import os
import shutil
import tempfile

import openpyxl


class ExcelToPDF(object):
    def __init__(self, kwargs={}):
        cmd_template = """
          soffice --convert-to
          "pdf:calc_pdf_Export:{\"SinglePageSheets\":{\"type\":\"boolean\",\"value\":\"true\"}}"
          --outdir
        """
        cmd_template2 = """
            soffice --headless --convert-to xlsx --outdir
        """

        cmd_template3 = 'sed -e \'s/\t/,/g\' "{0}" > "{1}"'

        def _norm_cmd(cmd):
            return " ".join([p.strip() for p in cmd.strip().split()])

        self.cmd_template = _norm_cmd(cmd_template)
        self.cmd_template2 = _norm_cmd(cmd_template2)
        self.cmd_template3 = cmd_template3

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
                cmd = self.cmd_template2 + ' "{1}" "{0}"'.format(input_file, temp_dir)
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

            cmd = self.cmd_template + ' "{1}" "{0}"'.format(input_file, temp_dir)
            ExcelToPDF.run(cmd)

            if output_file is not None:
                shutil.move(temp_output_file, output_file)

            if to_bytes:
                return open(temp_output_file, "rb").read()
