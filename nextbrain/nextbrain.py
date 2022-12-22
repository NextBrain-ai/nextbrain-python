import csv
import requests
import time

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

try:
    import aiohttp
    import asyncio
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False


class BaseNextBrain(ABC):

    def __init__(self, access_token: str, backend_url: str = 'https://api.nextbrain.ai'):
        self.access_token = access_token
        self.backend_url = backend_url

    @staticmethod
    def load_csv(path: str) -> List[List[Any]]:
        with open(path, 'r') as read_obj:
            return list(csv.reader(read_obj))

    @abstractmethod
    def wait_model(self, model_id: str, wait_imported: bool = True):
        raise NotImplementedError()

    @abstractmethod
    def upload_model(self, table: List[List[Any]]):
        raise NotImplementedError()

    @abstractmethod
    def train_model(self, model_id: str, target: str):
        raise NotImplementedError()

    @abstractmethod
    def predict_model(self, model_id: str, header: List[str], rows: List[List[Any]]) -> List[Any]:
        raise NotImplementedError()

    @abstractmethod
    def upload_and_predict(self, table: List[List[Any]], predict_table: List[List[Any]], target: str) -> Dict:
        raise NotImplementedError()


class NextBrain(BaseNextBrain):

    def __init__(self, access_token: str):
        super().__init__(access_token)

    def wait_model(self, model_id: str, wait_imported: bool = True) -> None:
        while True:
            response = requests.post(f'{self.backend_url}/model/status_token/{model_id}', json={
                'access_token': self.access_token,
            })
            data = response.json()
            if wait_imported:
                if data['dataset_status'] == 'imported':
                    return
                elif data['dataset_status'] == 'error':
                    raise Exception('Error importing model')

            else:
                if data['status'] == 'trained':
                    return
                elif data['status'] == 'error':
                    raise Exception('Error training model')

            time.sleep(2)

    def upload_model(self, table: List[List[Any]]) -> str:
        response = requests.post(f'{self.backend_url}/csv/import_matrix_token', json={
            'access_token': self.access_token,
            'matrix': table,
        })

        data = response.json()
        model_id = data['model']['id']
        self.wait_model(model_id)
        return model_id

    def train_model(self, model_id: str, target: str) -> None:
        requests.post(f'{self.backend_url}/model/train_token', json={
            'access_token': self.access_token,
            'target': target,
            'model_id': model_id,
        })
        self.wait_model(model_id, wait_imported=False)

    def predict_model(self, model_id: str, header: List[str], rows: List[List[Any]]) -> Dict:
        response = requests.post(f'{self.backend_url}/model/predict_token/{model_id}', json={
            'access_token': self.access_token,
            'header': header,
            'rows': rows,
        })
        return response.json()

    def upload_and_predict(self, table: List[List[Any]], predict_table: List[List[Any]], target: str) -> Tuple[str, Dict]:
        model_id = self.upload_model(table)
        self.train_model(model_id, target)
        return model_id, self.predict_model(model_id, predict_table[0], predict_table[1:])


class AsyncNextBrain(BaseNextBrain):

    def __init__(self, access_token: str):
        if not ASYNC_AVAILABLE:
            raise ImportError(
                'asyncio and aiohttp are required for async usage')
        super().__init__(access_token)

    async def wait_model(self, model_id: str, wait_imported: bool = True) -> None:
        json = {
            'access_token': self.access_token
        }
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.post(f'{self.backend_url}/model/status_token/{model_id}', json=json) as response:
                    data = await response.json()
                    if wait_imported:
                        if data['dataset_status'] == 'imported':
                            return
                        elif data['dataset_status'] == 'error':
                            raise Exception('Error importing model')

                    else:
                        if data['status'] == 'trained':
                            return
                        elif data['status'] == 'error':
                            raise Exception('Error training model')

                await asyncio.sleep(2)

    async def upload_model(self, table: List[List[Any]]) -> str:
        json = {
            'access_token': self.access_token,
            'matrix': table,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.backend_url}/csv/import_matrix_token', json=json) as response:
                data = await response.json()
                model_id = data['model']['id']
                await self.wait_model(model_id)
                return model_id

    async def train_model(self, model_id: str, target: str) -> None:
        json = {
            'access_token': self.access_token,
            'target': target,
            'model_id': model_id,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.backend_url}/model/train_token', json=json) as response:
                # Sequential await is required
                await response.json()
                await self.wait_model(model_id, wait_imported=False)

    async def predict_model(self, model_id: str, header: List[str], rows: List[List[Any]]) -> Dict:
        json = {
            'access_token': self.access_token,
            'header': header,
            'rows': rows,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.backend_url}/model/predict_token/{model_id}', json=json) as response:
                return await response.json()

    async def upload_and_predict(self, table: List[List[Any]], predict_table: List[List[Any]], target: str) -> Tuple[str, Dict]:
        # Required sequential await
        model_id = await self.upload_model(table)
        await self.train_model(model_id, target)
        return model_id, await self.predict_model(model_id, predict_table[0], predict_table[1:])
