# NextBrain
Convenient access to the [NextBrain](https://nextbrain.ai) API from python

## Installation
```bash
pip install nextbrain
```

If you want to use the async version you need to install `asyncio` and `aiohttp`:

```bash
pip install asyncio aiohttp
```

## Normal usage

### All steps in one.
```python
from nextbrain import NextBrain
from typing import Any, List

def main():
    nb = NextBrain('<YOUR-ACCESS-TOKEN-HERE>')

    # You can create your custom table and predict table by your own from any source
    table: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-TRAINING-CSV>')
    predict_table: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-PREDICTING-CSV>')

    model_id, response = nb.upload_and_predict(table, predict_table, '<YOUR-TARGET-COLUMN>')
    # model_id is also returned in order to predict multiple times against same model
    print(response)

if __name__ == '__main__':
    main()
```

### Step by step
```python
from nextbrain import NextBrain
from typing import Any, List

def main():
    nb = NextBrain('<YOUR-ACCESS-TOKEN-HERE>')

    # You can create your custom table and predict table by your own from any source
    table: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-TRAINING-CSV>')
    # Upload the model to NextBrain service
    model_id: str = nb.upload_model(table)
    # Train the model
    # You can re-train a previous model
    nb.train_model(model_id, '<YOUR-TARGET-COLUMN>')

    predict_table: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-PREDICTING-CSV>')
    # You can predict multiple using the same model (don't need to create a new model each time)
    response = nb.predict_model(model_id, predict_table[0], predict_table[1:])
    print(response)

if __name__ == '__main__':
    main()
```

## Async usage

### All steps in one.
```python
from nextbrain import AsyncNextBrain
from typing import Any, List

async def main():
    nb = AsyncNextBrain('<YOUR-ACCESS-TOKEN-HERE>')

    # You can create your custom table and predict table by your own from any source
    table: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-TRAINING-CSV>')
    predict_table: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-PREDICTING-CSV>')

    model_id, response = await nb.upload_and_predict(table, predict_table, '<YOUR-TARGET-COLUMN>')
    # model_id is also returned in order to predict multiple times against same model
    print(response)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```

### Step by step
```python
from nextbrain import AsyncNextBrain
from typing import Any, List

async def main():
    nb = AsyncNextBrain('<YOUR-ACCESS-TOKEN-HERE>')

    # You can create your custom table and predict table by your own from any source
    table: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-TRAINING-CSV>')
    # Upload the model to NextBrain service
    model_id: str = await nb.upload_model(table)
    # Train the model
    # You can re-train a previous model
    await nb.train_model(model_id, '<YOUR-TARGET-COLUMN>')

    predict_table: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-PREDICTING-CSV>')
    # You can predict multiple using the same model (don't need to create a new model each time)
    response = await nb.predict_model(model_id, predict_table[0], predict_table[1:])
    print(response)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```