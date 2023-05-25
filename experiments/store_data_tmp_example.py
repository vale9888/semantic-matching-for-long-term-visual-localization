from GSheet import GSheet


def main():
    gsheet = GSheet(worksheet_name='Experiments')
    values = ['Ciao', 'Valentina', 'prova', 1, 2.54, 'A', 'dopo']
    gsheet.store_row(values)


if __name__ == '__main__':
    main()
