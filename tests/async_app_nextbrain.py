from nextbrain import AsyncNextBrain


async def async_test():
    nb = AsyncNextBrain('YOUR-APP-KEY', is_app=True)
    table = nb.load_csv('../example_data/London_Water.csv')
    response = await nb.upload_and_predict(table, table, 'Consumption')
    print(str(response)[:1000])

if __name__ == '__main__':
    import asyncio
    asyncio.run(async_test())
