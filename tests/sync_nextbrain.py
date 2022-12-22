from nextbrain import NextBrain


def main():
    nb = NextBrain('37f3ea94a4311a9ccd96897324bee84e')
    matrix = nb.load_csv('../example_data/London_Water.csv')
    response = nb.upload_and_predict(matrix, matrix, 'Consumption')
    print(str(response)[:1000])


if __name__ == '__main__':
    main()
