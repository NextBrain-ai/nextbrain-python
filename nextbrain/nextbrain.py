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


class UnauthorizedException(Exception):

    def __init__(self, message: str = 'Unauthorized'):
        super().__init__(message)


class BaseNextBrain(ABC):

    def __init__(
        self,
        access_token: str,
        backend_url: str = 'https://api.nextbrain.ai',
        is_app: bool = False,
    ):
        self.access_token = access_token
        self.backend_url = backend_url
        self.is_app = is_app

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
    def train_model(self, model_id: str, target: str, is_lightning: bool = False):
        raise NotImplementedError()

    @abstractmethod
    def predict_model(self, model_id: str, header: List[str], rows: List[List[Any]]) -> List[Any]:
        raise NotImplementedError()

    @abstractmethod
    def random_predict(self, model_id: str) -> List[Any]:
        raise NotImplementedError()

    @abstractmethod
    def get_predict_columns(self, model_id: str) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def upload_and_predict(self, table: List[List[Any]], predict_table: List[List[Any]], target: str, is_lightning: bool = False) -> Dict:
        raise NotImplementedError()

    @abstractmethod
    def get_all_model_ids(self) -> List[Dict]:
        raise NotImplementedError()


class NextBrain(BaseNextBrain):

    def __init__(self, access_token: str, *args, **kwargs):
        super().__init__(access_token, *args, **kwargs)

    def get_accuracy(self, model_id: str) -> None:
        if self.is_app:
            response = requests.get(
                f'{self.backend_url}/app/acc/{model_id}',
                headers={
                    'access_token': self.access_token
                }
            )
        else:
            response = requests.post(f'{self.backend_url}/model/acc_token/{model_id}', json={
                'access_token': self.access_token,
            })

        if response.status_code == 401:
            raise UnauthorizedException()

        return response.json()

    def wait_model(self, model_id: str, wait_imported: bool = True) -> None:
        while True:
            if self.is_app:
                response = requests.post(
                    f'{self.backend_url}/app/status/{model_id}',
                    headers={
                        'access_token': self.access_token
                    }
                )
            else:
                response = requests.post(f'{self.backend_url}/model/status_token/{model_id}', json={
                    'access_token': self.access_token,
                })

            if response.status_code == 401:
                raise UnauthorizedException()

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
        if self.is_app:
            response = requests.post(
                f'{self.backend_url}/app/import_matrix',
                json={
                    'matrix': table,
                },
                headers={
                    'access_token': self.access_token
                }
            )
        else:
            response = requests.post(
                f'{self.backend_url}/csv/import_matrix_token',
                json={
                    'access_token': self.access_token,
                    'matrix': table,
                }
            )

        if response.status_code == 401:
            raise UnauthorizedException()

        data = response.json()
        model_id = data['model']['id']
        self.wait_model(model_id)
        return model_id

    def train_model(self, model_id: str, target: str, is_lightning: bool = False) -> None:
        if self.is_app:
            response = requests.post(
                f'{self.backend_url}/app/train',
                json={
                    'target': target,
                    'model_id': model_id,
                    'is_lightning': is_lightning,
                },
                headers={
                    'access_token': self.access_token
                }
            )
        else:
            response = requests.post(f'{self.backend_url}/model/train_token', json={
                'access_token': self.access_token,
                'target': target,
                'model_id': model_id,
                'is_lightning': is_lightning,
            })

        if response.status_code == 401:
            raise UnauthorizedException()

        self.wait_model(model_id, wait_imported=False)

    def predict_model(self, model_id: str, table: List[List[Any]]) -> Dict:
        header = table[0]
        rows = table[1:]
        if self.is_app:
            response = requests.post(
                f'{self.backend_url}/app/predict/{model_id}',
                json={
                    'header': header,
                    'rows': rows,
                },
                headers={
                    'access_token': self.access_token
                }
            )
        else:
            response = requests.post(f'{self.backend_url}/model/predict_token/{model_id}',
                                     json={
                                         'access_token': self.access_token,
                                         'header': header,
                                         'rows': rows,
                                     }
                                     )

        if response.status_code == 401:
            raise UnauthorizedException()

        return response.json()

    def random_predict(self, model_id: str) -> Dict:
        if self.is_app:
            response = requests.post(
                f'{self.backend_url}/app/random_predict/{model_id}',
                headers={
                    'access_token': self.access_token
                }
            )
        else:
            response = requests.post(
                f'{self.backend_url}/model/random_predict_token/{model_id}',
                json={
                    'access_token': self.access_token,
                }
            )

        if response.status_code == 401:
            raise UnauthorizedException()

        return response.json()

    def get_predict_columns(self, model_id: str) -> List[str]:
        if self.is_app:
            response = requests.get(
                f'{self.backend_url}/app/predict_columns/{model_id}',
                headers={
                    'access_token': self.access_token
                }
            )
        else:
            response = requests.post(
                f'{self.backend_url}/model/predict_columns_token/{model_id}',
                json={
                    'access_token': self.access_token,
                }
            )

        if response.status_code == 401:
            raise UnauthorizedException()

        return response.json()

    def upload_and_predict(self, table: List[List[Any]], predict_table: List[List[Any]], target: str, is_lightning: bool = False) -> Tuple[str, Dict]:
        model_id = self.upload_model(table)
        self.train_model(model_id, target, is_lightning=is_lightning)
        return model_id, self.predict_model(model_id, predict_table)

    def delete_model(self, model_id: str) -> None:
        if self.is_app:
            response = requests.post(
                f'{self.backend_url}/app/delete_model/{model_id}',
                headers={
                    'access_token': self.access_token
                }
            )
        else:
            response = requests.post(f'{self.backend_url}/model/delete_model_token/{model_id}',
                                     json={
                                         'access_token': self.access_token,
                                     }
                                     )

        if response.status_code == 401:
            raise UnauthorizedException()

    def get_all_model_ids(self) -> List[Dict]:
        if self.is_app:
            response = requests.get(
                f'{self.backend_url}/app/model_ids',
                headers={
                    'access_token': self.access_token
                }
            )
        else:
            response = requests.post(
                f'{self.backend_url}/model/model_ids_token',
                json={
                    'access_token': self.access_token,
                }
            )

        if response.status_code == 401:
            raise UnauthorizedException()

        return response.json()


class AsyncNextBrain(BaseNextBrain):

    def __init__(self, access_token: str, *args, **kwargs):
        if not ASYNC_AVAILABLE:
            raise ImportError(
                'asyncio and aiohttp are required for async usage')
        super().__init__(access_token, *args, **kwargs)

    async def get_accuracy(self, model_id: str) -> float:
        async with aiohttp.ClientSession() as session:
            if self.is_app:
                response = await session.get(
                    f'{self.backend_url}/app/acc/{model_id}',
                    headers={
                        'access_token': self.access_token
                    }
                )
            else:
                response = await session.post(f'{self.backend_url}/model/acc_token/{model_id}', json={
                    'access_token': self.access_token,
                })

            if response.status == 401:
                raise UnauthorizedException()

            return await response.json()

    async def wait_model(self, model_id: str, wait_imported: bool = True) -> None:
        async with aiohttp.ClientSession() as session:
            while True:
                kwargs = {}
                if self.is_app:
                    url = f'{self.backend_url}/app/status/{model_id}'
                    kwargs['headers'] = {
                        'access_token': self.access_token
                    }
                else:
                    url = f'{self.backend_url}/model/status_token/{model_id}'
                    kwargs['json'] = {
                        'access_token': self.access_token
                    }

                async with session.post(url, **kwargs) as response:
                    if response.status == 401:
                        raise UnauthorizedException()
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
        kwargs = {}
        if self.is_app:
            url = f'{self.backend_url}/app/import_matrix'
            kwargs['json'] = {
                'matrix': table,
            }
            kwargs['headers'] = {
                'access_token': self.access_token
            }
        else:
            url = f'{self.backend_url}/csv/import_matrix_token'
            kwargs['json'] = {
                'matrix': table,
                'access_token': self.access_token
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, **kwargs) as response:
                data = await response.json()
                model_id = data['model']['id']
                await self.wait_model(model_id)
                return model_id

    async def train_model(self, model_id: str, target: str, is_lightning: bool = False) -> None:
        json = {
            'target': target,
            'model_id': model_id,
            'is_lightning': is_lightning,
        }

        if self.is_app:
            url = f'{self.backend_url}/app/train'
            kwargs = {
                'json': json,
                'headers': {
                    'access_token': self.access_token
                }
            }
        else:
            url = f'{self.backend_url}/model/train_token'
            json['access_token'] = self.access_token
            kwargs = {
                'json': json
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, **kwargs) as response:
                # Sequential await is required
                await response.json()
                await self.wait_model(model_id, wait_imported=False)

    async def predict_model(self, model_id: str, table: List[List[Any]]) -> Dict:
        header = table[0]
        rows = table[1:]
        json = {
            'header': header,
            'rows': rows,
        }

        if self.is_app:
            url = f'{self.backend_url}/app/predict/{model_id}'
            kwargs = {
                'json': json,
                'headers': {
                    'access_token': self.access_token
                }
            }
        else:
            url = f'{self.backend_url}/model/predict_token/{model_id}'
            json['access_token'] = self.access_token
            kwargs = {
                'json': json
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, **kwargs) as response:
                return await response.json()

    async def random_predict(self, model_id: str) -> Dict:
        if self.is_app:
            url = f'{self.backend_url}/app/random_predict/{model_id}'
            kwargs = {
                'headers': {
                    'access_token': self.access_token
                }
            }
        else:
            url = f'{self.backend_url}/model/random_predict_token/{model_id}'
            kwargs = {
                'json': {
                    'access_token': self.access_token
                }
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, **kwargs) as response:
                return await response.json()

    async def get_predict_columns(self, model_id: str) -> List[str]:
        if self.is_app:
            url = f'{self.backend_url}/app/predict_columns/{model_id}'
            kwargs = {
                'headers': {
                    'access_token': self.access_token
                }
            }
        else:
            url = f'{self.backend_url}/model/predict_columns_token/{model_id}'
            kwargs = {
                'json': {
                    'access_token': self.access_token
                }
            }

        async with aiohttp.ClientSession() as session:
            method = session.get if self.is_app else session.post
            async with method(url, **kwargs) as response:
                return await response.json()

    async def upload_and_predict(self, table: List[List[Any]], predict_table: List[List[Any]], target: str, is_lightning: bool = False) -> Tuple[str, Dict]:
        # Required sequential await
        model_id = await self.upload_model(table)
        await self.train_model(model_id, target, is_lightning=is_lightning)
        return model_id, await self.predict_model(model_id, predict_table)

    async def delete_model(self, model_id: str) -> None:
        if self.is_app:
            url = f'{self.backend_url}/app/delete_model/{model_id}'
            kwargs = {
                'headers': {
                    'access_token': self.access_token
                }
            }
        else:
            url = f'{self.backend_url}/model/delete_model_token/{model_id}'
            kwargs = {
                'json': {
                    'access_token': self.access_token,
                }
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, **kwargs) as response:
                if response.status == 401:
                    raise UnauthorizedException()
                # Sequential await is required
                await response.json()

    async def get_all_model_ids(self) -> List[Dict]:
        if self.is_app:
            url = f'{self.backend_url}/app/model_ids'
            kwargs = {
                'headers': {
                    'access_token': self.access_token
                }
            }
        else:
            url = f'{self.backend_url}/model/model_ids_token'
            kwargs = {
                'json': {
                    'access_token': self.access_token
                }
            }

        async with aiohttp.ClientSession() as session:
            method = session.get if self.is_app else session.post
            async with method(url, **kwargs) as response:
                if response.status == 401:
                    raise UnauthorizedException()
                return await response.json()
