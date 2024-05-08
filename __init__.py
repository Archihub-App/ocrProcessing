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
from app.api.users.services import has_role
from app.api.tasks.services import add_task
from dotenv import load_dotenv
import pdfplumber
load_dotenv()

mongodb = DatabaseHandler.DatabaseHandler()
WEB_FILES_PATH = os.environ.get('WEB_FILES_PATH', '')
ORIGINAL_FILES_PATH = os.environ.get('ORIGINAL_FILES_PATH', '')
plugin_path = os.path.dirname(os.path.abspath(__file__))
models_path = plugin_path + '/models'
tessdata_path = plugin_path + '/tessdata'

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
        if not body['overwrite']:
            records_filters['processing.ocrProcessing'] = {'$exists': False}

        records = list(mongodb.get_all_records('records', records_filters, fields={
            '_id': 1, 'mime': 1, 'filepath': 1, 'processing': 1}))

        if len(records) > 0:
            model = lp.Detectron2LayoutModel(models_path + '/config_1.yaml',
                                             models_path + '/mymodel_1.pth',
                                             extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7, "MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER", 0.5],
                                             label_map={0: "Figure", 1: "Footnote", 2: "List", 3: "Table", 4: "Text", 5: "Title"})
            
            ocr_agent = lp.TesseractAgent(languages='spa')

            for record in records:

                path = WEB_FILES_PATH + '/' + \
                    record['processing']['fileProcessing']['path'] + '/web/big'
                path_original = ORIGINAL_FILES_PATH + '/' + \
                    record['processing']['fileProcessing']['path'] + '.pdf'
                
                files = os.listdir(path)
                page = 0
                resp = []

                

                for f in files:
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
                    footnote_blocks = lp.Layout(
                        [b for b in layout if b.type == 'Footnote'])

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
                    

                    for b in text_blocks:
                        txt = ''
                        segment_words = []

                        if not has_text:
                            txt = segment_image(b, image)
                        else:
                            segment_words = extract_segment_words(words, b)
                            for w in segment_words:
                                txt += w['text'] + ' '

                        obj = get_obj(b, txt, 'text', segment_words)

                        resp_page.append(obj)

                    for b in title_blocks:
                        txt = ''
                        segment_words = []

                        if not has_text:
                            txt = segment_image(b, image)
                        else:
                            segment_words = extract_segment_words(words, b)
                            for w in segment_words:
                                txt += w['text'] + ' '

                        obj = get_obj(b, txt, 'title', segment_words)

                        resp_page.append(obj)

                    for b in list_blocks:
                        txt = ''
                        segment_words = []

                        if not has_text:
                            txt = segment_image(b, image)
                        else:
                            segment_words = extract_segment_words(words, b)
                            for w in segment_words:
                                txt += w['text'] + ' '

                        obj = get_obj(b, txt, 'list', segment_words)

                        resp_page.append(obj)

                    for b in table_blocks:
                        txt = ''
                        segment_words = []

                        if not has_text:
                            txt = segment_image(b, image)
                        else:
                            segment_words = extract_segment_words(words, b)
                            for w in segment_words:
                                txt += w['text'] + ' '

                        obj = get_obj(b, txt, 'table', segment_words)

                        resp_page.append(obj)

                    for b in footnote_blocks:
                        txt = ''
                        segment_words = []

                        if not has_text:
                            txt = segment_image(b, image)
                        else:
                            segment_words = extract_segment_words(words, b)
                            for w in segment_words:
                                txt += w['text'] + ' '

                        obj = get_obj(b, txt, 'footnote', segment_words)

                        resp_page.append(obj)

                    for b in figure_blocks:
                        obj = {
                            'type': 'figure',
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
                else:
                    return self.settings['settings_' + type]
            except Exception as e:
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

                print(data)

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
        ]
    }
}
