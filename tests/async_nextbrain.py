from nextbrain import AsyncNextBrain


async def async_test():
    nb = AsyncNextBrain('37f3ea94a4311a9ccd96897324bee84e')
    matrix = nb.load_csv('../example_data/London_Water.csv')
    response = await nb.upload_and_predict(matrix, matrix, 'Consumption')
    print(str(response)[:1000])

if __name__ == '__main__':
    import asyncio
    asyncio.run(async_test())
