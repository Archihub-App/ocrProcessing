from app.utils.PluginClass import PluginClass
from flask_jwt_extended import jwt_required, get_jwt_identity
from celery import shared_task
from app.utils import DatabaseHandler
from flask import request
import os
import layoutparser as lp
import cv2
from app.api.records.models import RecordUpdate
from app.api.resources.services import update_cache as update_cache_resources
from app.api.records.services import update_cache as update_cache_records
from dotenv import load_dotenv
load_dotenv()

mongodb = DatabaseHandler.DatabaseHandler()
WEB_FILES_PATH = os.environ.get('WEB_FILES_PATH', '')
plugin_path = os.path.dirname(os.path.abspath(__file__))
models_path = plugin_path + '/models'


class ExtendedPluginClass(PluginClass):
    def __init__(self, path, import_name, name, description, version, author, type, settings):
        super().__init__(path, __file__, import_name, name,
                         description, version, author, type, settings)
        
    def add_routes(self):
        @self.route('/bulk', methods=['POST'])
        @jwt_required()
        def create_inventory():
            current_user = get_jwt_identity()
            body = request.get_json()

            if 'post_type' not in body:
                return {'msg': 'No se especificó el tipo de contenido'}, 400

            if not self.has_role('admin', current_user) and not self.has_role('processing', current_user):
                return {'msg': 'No tiene permisos suficientes'}, 401

            task = self.bulk.delay(body, current_user)
            self.add_task_to_user(
                task.id, 'ocrProcessing.bulk', current_user, 'msg')

            return {'msg': 'Se agregó la tarea a la fila de procesamientos'}, 201

    @shared_task(ignore_result=False, name='ocrProcessing.bulk')
    def bulk(body, user):
        filters = {
            'post_type': body['post_type']
        }

        if 'parent' in body:
            if body['parent']:
                filters['parents.id'] = body['parent']

        # obtenemos los recursos
        resources = list(mongodb.get_all_records(
            'resources', filters, fields={'_id': 1}))
        resources = [str(resource['_id']) for resource in resources]

        records_filters = {
            'parent.id': {'$in': resources},
            'processing.fileProcessing': {'$exists': True},
            '$or': [{'processing.fileProcessing.type': 'document'}]
        }
        if body['overwrite']:
            records_filters['processing.ocrProcessing'] = {'$exists': True}
        else:
            records_filters['processing.ocrProcessing'] = {
                '$exists': False}

        records = list(mongodb.get_all_records('records', records_filters, fields={
            '_id': 1, 'mime': 1, 'filepath': 1, 'processing': 1}))

        if len(records) > 0:
            model = lp.Detectron2LayoutModel('/home/nestor/.torch/iopath_cache/s/57zjbwv6gh3srry/config.yaml',
                                             '/home/nestor/.torch/iopath_cache/s/57zjbwv6gh3srry/model_final.pth',
                                             label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})
            ocr_agent = lp.TesseractAgent(languages='spa')

            for record in records:

                path = WEB_FILES_PATH + '/' + \
                    record['processing']['fileProcessing']['path'] + '/web/big'
                files = os.listdir(path)
                page = 0
                resp = []
                for f in files:
                    page += 1
                    image = cv2.imread(path + '/' + f)
                    image = image[..., ::-1]

                    layout = model.detect(image)

                    text_blocks = lp.Layout(
                        [b for b in layout if b.type == 'Text'])
                    title_blocks = lp.Layout(
                        [b for b in layout if b.type == 'Title'])
                    list_blocks = lp.Layout(
                        [b for b in layout if b.type == 'List'])
                    table_blocks = lp.Layout(
                        [b for b in layout if b.type == 'Table'])
                    figure_blocks = lp.Layout(
                        [b for b in layout if b.type == 'Figure'])

                    resp_page = []
                    for b in text_blocks:
                        segment_image = (
                            b.pad(left=5, right=5, top=5,
                                  bottom=5).crop_image(image)
                        )

                        segment_text = ocr_agent.detect(
                            segment_image, return_response=True, return_only_text=False)
                        obj = {
                            'text': segment_text['text'],
                            'type': 'text',
                            'bbox': {
                                'x': b.block.x_1,
                                'y': b.block.y_1,
                                'width': b.block.x_2 - b.block.x_1,
                                'height': b.block.y_2 - b.block.y_1
                            }
                        }

                        resp_page.append(obj)

                    for b in title_blocks:
                        segment_image = (
                            b.pad(left=5, right=5, top=5,
                                  bottom=5).crop_image(image)
                        )

                        segment_text = ocr_agent.detect(
                            segment_image, return_response=True, return_only_text=False)

                        obj = {
                            'text': segment_text['text'],
                            'type': 'title',
                            'bbox': {
                                'x': b.block.x_1,
                                'y': b.block.y_1,
                                'width': b.block.x_2 - b.block.x_1,
                                'height': b.block.y_2 - b.block.y_1
                            }
                        }

                        resp_page.append(obj)

                    for b in list_blocks:
                        segment_image = (
                            b.pad(left=5, right=5, top=5,
                                  bottom=5).crop_image(image)
                        )

                        segment_text = ocr_agent.detect(
                            segment_image, return_response=True, return_only_text=False)

                        obj = {
                            'text': segment_text['text'],
                            'type': 'list',
                            'bbox': {
                                'x': b.block.x_1,
                                'y': b.block.y_1,
                                'width': b.block.x_2 - b.block.x_1,
                                'height': b.block.y_2 - b.block.y_1
                            }
                        }

                        resp_page.append(obj)

                    for b in table_blocks:
                        segment_image = (
                            b.pad(left=5, right=5, top=5,
                                  bottom=5).crop_image(image)
                        )

                        segment_text = ocr_agent.detect(
                            segment_image, return_response=True, return_only_text=False)

                        obj = {
                            'text': segment_text['text'],
                            'type': 'table',
                            'bbox': {
                                'x': b.block.x_1,
                                'y': b.block.y_1,
                                'width': b.block.x_2 - b.block.x_1,
                                'height': b.block.y_2 - b.block.y_1
                            }
                        }

                        resp_page.append(obj)

                    for b in figure_blocks:
                        obj = {
                            'type': 'figure',
                            'bbox': {
                                'x': b.block.x_1,
                                'y': b.block.y_1,
                                'width': b.block.x_2 - b.block.x_1,
                                'height': b.block.y_2 - b.block.y_1
                            }
                        }

                        resp_page.append(obj)


                    resp.append({
                        'page': page,
                        'blocks': resp_page
                    })

                update = {
                    'processing': record['processing']
                }

                update['processing']['ocrProcessing'] = {
                    'type': 'lt_extraction',
                    'result': resp
                }

                update = RecordUpdate(**update)
                mongodb.update_record(
                    'records', {'_id': record['_id']}, update)

        update_cache_records()
        update_cache_resources()
        return 'Extracción de texto finalizada'


plugin_info = {
    'name': 'Extraer texto de documentos',
    'description': 'Plugin para extraer texto de documentos en el gestor documental',
    'version': '0.1',
    'author': 'Néstor Andrés Peña',
    'type': ['bulk'],
    'settings': {
        'settings_bulk': [
            {
                'type':  'instructions',
                'title': 'Instrucciones',
                'text': 'Este plugin segmenta los documentos y solo extrae texto de los bloques de texto importantes.',
            },
            {
                'type': 'checkbox',
                'label': 'Sobreescribir procesamientos existentes',
                'id': 'overwrite',
                'default': False,
                'required': False,
            },
            {
                'type': 'select',
                'label': 'Modelo',
                'id': 'model',
                'default': 'model_final.pth',
                'options': [
                    {'value': 'model_final.pth', 'label': 'Por defecto'},
                ],
                'required': False,
            }
        ]
    }
}
