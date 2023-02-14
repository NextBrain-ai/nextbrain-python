from nextbrain import NextBrain


def main():
    nb = NextBrain('YOUR-APP-KEY', is_app=True)
    table = nb.load_csv('../example_data/London_Water.csv')
    response = nb.upload_and_predict(table, table, 'Consumption')
    print(str(response)[:1000])


if __name__ == '__main__':
    main()
