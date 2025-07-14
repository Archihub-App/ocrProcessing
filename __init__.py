from app.utils.PluginClass import PluginClass
from flask_jwt_extended import jwt_required, get_jwt_identity
from celery import shared_task
from app.utils import DatabaseHandler
from flask import request
import os
from app.api.records.models import RecordUpdate
from app.api.resources.services import update_cache as update_cache_resources
from app.api.records.services import update_cache as update_cache_records
from app.api.users.services import has_role
from app.api.tasks.services import add_task
from dotenv import load_dotenv
from bson.objectid import ObjectId
import json
import pdfplumber
import importlib 
from datetime import datetime                                                   

load_dotenv()

mongodb = DatabaseHandler.DatabaseHandler()
WEB_FILES_PATH = os.environ.get('WEB_FILES_PATH', '')
ORIGINAL_FILES_PATH = os.environ.get('ORIGINAL_FILES_PATH', '')
plugin_path = os.path.dirname(os.path.abspath(__file__))
models_path = plugin_path + '/models'
tessdata_path = plugin_path + '/tessdata'

class ExtendedPluginClass(PluginClass):
    def __init__(self, path, import_name, name, description, version, author, type, settings, actions=None, capabilities=None, **kwargs):
        super().__init__(path, __file__, import_name, name, description, version, author, type, settings, actions=actions, capabilities=capabilities, **kwargs)
        
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
        
        @self.route('/blockProcessing', methods=['POST'])
        @jwt_required()
        def block_processing():
            current_user = get_jwt_identity()
            body = request.get_json()
            
            print(body)

            task = self.bulk.delay(body, current_user)
            self.add_task_to_user(task.id, 'ocrProcessing.blockProcessing', current_user, 'msg')
            

            return {'msg': 'Se agregó la tarea a la fila de procesamientos'}, 201

    @shared_task(ignore_result=False, name='ocrProcessing.bulk')
    def bulk(body, user):       
        import cv2
        import layoutparser as lp

        def check_text_extraction(pdf_path, page):
            with pdfplumber.open(pdf_path) as pdf:
                page_ = pdf.pages[page]
                text = ""
                text += page_.extract_text()
            
            return len(text) > 0
        
        def extract_words(pdf_path, page):
            with pdfplumber.open(pdf_path) as pdf:
                page_ = pdf.pages[page]
                words = page_.extract_words()
                
            return words, page_.width, page_.height
        
        def extract_words_bbox(words, bbox, page_w, page_h):
            resp = []
            for word in words:
                if word['x0'] / page_w >= bbox['x_1'] and word['x1'] / page_w <= bbox['x_2'] and word['top'] / page_h >= bbox['y_1'] and word['bottom'] / page_h <= bbox['y_2']:
                    resp.append(word)
            return resp

        if 'records' not in body:
            filters = {
                'post_type': body['post_type']
            }

            if body['parent'] and len(body['resources']) == 0:
                filters = {'$or': [{'parents.id': body['parent'], 'post_type': body['post_type']}, {'_id': ObjectId(body['parent'])}], **filters}
            
            if body['resources']:
                if len(body['resources']) > 0:
                    filters = {'_id': {'$in': [ObjectId(resource) for resource in body['resources']]}, **filters}

            # obtenemos los recursos
            resources = list(mongodb.get_all_records(
                'resources', filters, fields={'_id': 1}))
            records_filters = {
                'parent.id': {'$in': resources},
                'processing.fileProcessing': {'$exists': True},
                '$or': [{'processing.fileProcessing.type': 'document'}]
            }
            if not body['overwrite']:
                records_filters['processing.ocrProcessing'] = {'$exists': False}
                
        else:
            records_filters = {'_id': {'$in': [ObjectId(record) for record in body['records']]}}
            
        records = list(mongodb.get_all_records('records', records_filters, fields={
            '_id': 1, 'mime': 1, 'filepath': 1, 'processing': 1}))

        if len(records) > 0:
            page_block = body.get('model', '*') == '*'
            page_only = body.get('page_only', False)
            page_to_process = body.get('opts', None)
            page_to_process = page_to_process.get('page', 1) if page_only else None
            
            if not page_block:
                label_map = importlib.import_module(f'.models.{body["model"]}.label_map', package=__name__)
                label_map = label_map.list_map[0]
                model = lp.Detectron2LayoutModel(os.path.join(models_path, body['model'], 'config.yaml'),
                                                os.path.join(models_path, body['model'], 'model.pth'),
                                                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7, "MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER", 0.5, "MODEL.DEVICE", "cpu"],
                                                label_map=label_map, device='cpu')
            
                ocr_agent = lp.TesseractAgent(languages='spa')
            else:
                label_map = ['Page']

            
            ocr_types = body.get('ocr_types', list(label_map.values()))
            for record in records:
                path = os.path.join(WEB_FILES_PATH, record['processing']['fileProcessing']['path'], 'web', 'big')
                path_original = os.path.join(ORIGINAL_FILES_PATH, record['processing']['fileProcessing']['path'] + '.pdf')
                files = sorted(os.listdir(path))
                page = 0
                resp = []
                
                for i, f in enumerate(files):
                    if page_only and page + 1 != page_to_process:
                        page += 1
                        continue
                    
                    words = None
                    image = cv2.imread(path + '/' + f)
                    image = image[..., ::-1]
                    image_width = image.shape[1]
                    image_height = image.shape[0]
                    aspect_ratio = image_width / image_height

                    has_text = check_text_extraction(path_original, page)
                    if has_text:
                        words, w_doc, h_doc = extract_words(path_original, page)

                    page += 1

                    layout = model.detect(image)

                    blocks = []
                    for l in label_map.values():
                        if not page_block:
                            _ = lp.Layout([b for b in layout if b.type == l])
                            blocks.append(_)
                        else:
                            if l == 'Page':
                                _ = {
                                    'block': {
                                        'x_1': 0,
                                        'y_1': 0,
                                        'x_2': image_width,
                                        'y_2': image_height
                                    },
                                    'type': l
                                }
                                blocks.append(_)

                    resp_page = []

                    def segment_image(b, image):
                        segment_image = (
                            b.pad(left=5, right=5, top=5,
                                bottom=5).crop_image(image)
                        )

                        segment_text = ocr_agent.detect(
                            segment_image, return_response=True, return_only_text=False)
                        
                        txt = segment_text['text']

                        return txt
                    
                    def extract_segment_words(words, b):
                        segment_words = extract_words_bbox(words, {
                            'x_1': (b.block.x_1 - 50) / image_width,
                            'y_1': (b.block.y_1 - 50) / image_height,
                            'x_2': (b.block.x_2 + 50) / image_width,
                            'y_2': (b.block.y_2 + 50) / image_height
                        }, w_doc, h_doc)

                        return segment_words
                    
                    def get_obj(b, txt, type, segment_words):
                        obj = {
                            'text': txt,
                            'type': type,
                            'bbox': {
                                'x': b.block.x_1 / image_width,
                                'y': b.block.y_1 / image_height,
                                'width': (b.block.x_2 - b.block.x_1) / image_width,
                                'height': (b.block.y_2 - b.block.y_1) / image_height
                            },
                            'words': [{
                                'text': s['text'],
                                'bbox': {
                                    'x': s['x0'] / w_doc,
                                    'y': s['top'] / h_doc,
                                    'width': (s['x1'] - s['x0']) / w_doc,
                                    'height': (s['bottom'] - s['top']) / h_doc
                                }
                            } for s in segment_words]
                        }
                        return obj
                    
                    for block in blocks:
                        for b in block:
                            if b.type in ocr_types:
                                txt = ''
                                segment_words = []

                                if not has_text:
                                    txt = segment_image(b, image)
                                else:
                                    segment_words = extract_segment_words(words, b)
                                    for w in segment_words:
                                        txt += w['text'] + ' '

                                obj = get_obj(b, txt, b.type, segment_words)

                                resp_page.append(obj)
                            else:
                                obj = {
                                    'type': b.type,
                                    'bbox': {
                                        'x': b.block.x_1 / image_width,
                                        'y': b.block.y_1 / image_height,
                                        'width': (b.block.x_2 - b.block.x_1) / image_width,
                                        'height': (b.block.y_2 - b.block.y_1) / image_height
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

                string_label_map = {str(k): v for k, v in label_map.items()}

                update['processing']['ocrProcessing'] = {
                    'type': 'lt_extraction',
                    'model': body['model'],
                    'labels': string_label_map,
                    'result': resp
                }
                update['updatedAt'] = datetime.now()
                update['updatedBy'] = user if user else 'system'

                update = RecordUpdate(**update)
                mongodb.update_record(
                    'records', {'_id': record['_id']}, update)

        instance = ExtendedPluginClass('ocrProcessing','', **plugin_info)
        instance.clear_cache()
        return 'Extracción de texto finalizada'

    def get_actions(self):
        for a in self.actions:
            if 'placement' in a:
                if a['placement'] == 'detail_record':
                    form = a.get('extraOpts', [])
                    
                    template_folders = os.listdir(models_path)
                    template_folders = [t for t in template_folders if os.path.isdir(os.path.join(models_path, t))]
                    template_folders = [t for t in template_folders if t != '__pycache__']
                    
                    model_block = {
                        'type': 'select',
                        'id': 'model',
                        'label': 'Modelo de segmentación',
                        'default': '*',
                        'options': [{'value': '*', 'label': 'Detectar un único bloque en toda la página'}, {'value': 'existing', 'label': 'Usar los bloques guardados en el sistema'}] + [{'value': t, 'label': t} for t in template_folders],
                    }
                    
                    condi_block = {
                        'type': 'condition',
                        'id': 'ocr_types',
                        'label': 'Tipos de segmentación para OCR',
                        'default': [],
                        'id_condition': 'model',
                        'condition': '==',
                        'options': [],
                    }
                    for folder in template_folders:
                        label_map = importlib.import_module(f'.models.{folder}.label_map', package=__name__)
                        label_map = label_map.list_map[0]
                        label_map = [{'label': label_map[key], 'value': label_map[key]} for key in label_map]

                        condi_block['options'].append({
                            'value': folder,
                            'fields': [
                                {
                                    'type': 'multiple-checkbox',
                                    'label': '',
                                    'id': folder + '_ocrtypes',
                                    'default': [],
                                    'required': False,
                                    'options': label_map
                                }
                            ]
                        })
                        
                    form = [model_block] + [condi_block] + form
                    
                    a['extraOpts'] = form
                    
        return self.actions

    def get_settings(self):
        @self.route('/settings/<type>', methods=['GET'])
        @jwt_required()
        def get_settings(type):
            try:
                current_user = get_jwt_identity()

                if not has_role(current_user, 'admin') and not has_role(current_user, 'processing'):
                    return {'msg': 'No tiene permisos suficientes'}, 401
                
                if type == 'all':
                    return self.settings
                elif type == 'settings':
                    return self.settings['settings']
                elif type == 'bulk':
                    # obtenemos todos los directorios en models_path
                    template_folders = os.listdir(models_path)
                    template_folders = [t for t in template_folders if os.path.isdir(os.path.join(models_path, t))]
                    template_folders = [t for t in template_folders if t != '__pycache__']

                    resp = [*self.settings['settings_bulk']]
                    resp.append({
                        'type': 'select',
                        'id': 'model',
                        'label': 'Modelo de segmentación',
                        'default': '',
                        'options': [{'value': t, 'label': t} for t in template_folders],
                        'required': True
                    })

                    condi_block = {
                        'type': 'condition',
                        'id': 'ocr_types',
                        'label': 'Tipos de segmentación para OCR',
                        'default': [],
                        'id_condition': 'model',
                        'condition': '==',
                        'options': [],
                    }
                    for folder in template_folders:
                        label_map = importlib.import_module(f'.models.{folder}.label_map', package=__name__)
                        label_map = label_map.list_map[0]
                        label_map = [{'label': label_map[key], 'value': label_map[key]} for key in label_map]

                        condi_block['options'].append({
                            'value': folder,
                            'fields': [
                                {
                                    'type': 'multiple-checkbox',
                                    'label': '',
                                    'id': folder + '_ocrtypes',
                                    'default': [],
                                    'required': False,
                                    'options': label_map
                                }
                            ]
                        })

                    resp.append(condi_block)
                    return resp
                elif type == 'block':
                    return self.settings['settings_block']
                else:
                    return self.settings['settings_' + type]
            except Exception as e:
                print(str(e))
                return {'msg': str(e)}, 500
            
        @self.route('/settings', methods=['POST'])
        @jwt_required()
        def set_settings_update():
            try:
                current_user = get_jwt_identity()

                if not has_role(current_user, 'admin') and not has_role(current_user, 'processing'):
                    return {'msg': 'No tiene permisos suficientes'}, 401
                
                body = request.form.to_dict()
                data = body['data']
                data = json.loads(data)

                self.set_plugin_settings(data)
                return {'msg': 'Configuración guardada'}, 200
            
            except Exception as e:
                return {'msg': str(e)}, 500

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
            }
        ],
        'settings_blocks': [
            {
                'type': 'select',
                'label': 'Tipo de bloque',
                'id': 'type',
                'default': '',
                'required': True,
            }
        ],
        'settings_words': [
            {
                'type': 'text',
                'label': 'Texto en el bloque',
                'id': 'text',
                'default': '',
                'required': False
            }
        ],
    },
    'actions': [
        {
            'placement': 'detail_record',
            'record_type': ['document'],
            'label': 'Detectar bloques de texto',
            'roles': ['admin', 'processing', 'editor'],
            'endpoint': 'blockProcessing',
            'icon': 'BorderOuter,TextFields',
            'extraOpts': [
                {
                    'type': 'checkbox',
                    'label': 'Solo procesar la página actual',
                    'id': 'page_only',
                    'default': False,
                    'required': False,
                },
            ]
        },
        
    ]
}
