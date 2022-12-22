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

    # You can create your custom matrix and predict matrix by your own from any source
    matrix: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-TRAINING-CSV>')
    predict_matrix: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-PREDICTING-CSV>')

    model_id, response = nb.upload_and_predict(matrix, predict_matrix, '<YOUR-TARGET-COLUMN>')
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

    # You can create your custom matrix and predict matrix by your own from any source
    matrix: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-TRAINING-CSV>')
    # Upload the model to NextBrain service
    model_id: str = nb.upload_model(matrix)
    # Train the model
    # You can re-train a previous model
    nb.train_model(model_id, '<YOUR-TARGET-COLUMN>')

    predict_matrix: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-PREDICTING-CSV>')
    # You can predict multiple using the same model (don't need to create a new model each time)
    response = nb.predict_model(model_id, predict_matrix[0], predict_matrix[1:])
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

    # You can create your custom matrix and predict matrix by your own from any source
    matrix: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-TRAINING-CSV>')
    predict_matrix: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-PREDICTING-CSV>')

    model_id, response = await nb.upload_and_predict(matrix, predict_matrix, '<YOUR-TARGET-COLUMN>')
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

    # You can create your custom matrix and predict matrix by your own from any source
    matrix: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-TRAINING-CSV>')
    # Upload the model to NextBrain service
    model_id: str = await nb.upload_model(matrix)
    # Train the model
    # You can re-train a previous model
    await nb.train_model(model_id, '<YOUR-TARGET-COLUMN>')

    predict_matrix: List[List[Any]] = nb.load_csv('<PATH-TO-YOUR-PREDICTING-CSV>')
    # You can predict multiple using the same model (don't need to create a new model each time)
    response = await nb.predict_model(model_id, predict_matrix[0], predict_matrix[1:])
    print(response)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```