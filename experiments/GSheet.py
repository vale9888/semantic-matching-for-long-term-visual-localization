# pip install gspread
# Guide: https://docs.gspread.org/en/v5.7.1/oauth2.html#service-account

import gspread
from gspread.exceptions import APIError


class GSheet:
    def __init__(self,
                 worksheet_name: str,
                 credentials: str = 'creds.json',
                 spreadsheet: str = 'SemanticMatchingExperiments'):
        gc = gspread.service_account(filename=credentials)
        sh = gc.open(spreadsheet)

        try:  # the worksheet already exists
            self.wks = sh.add_worksheet(title=worksheet_name, rows=10000, cols=52)
        except APIError:  # create new worksheet
            self.wks = sh.worksheet(worksheet_name)

    def store_row(self, values: list):
        self.wks.append_rows(values=[values])



