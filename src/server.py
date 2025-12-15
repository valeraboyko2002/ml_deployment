import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class MLRequestHandler(BaseHTTPRequestHandler):
    """Кастомный обработчик HTTP запросов для ML API"""
    
    def do_GET(self):
        """Обработка GET запросов"""
        if self.path == '/health':
            self._send_response(200, {'status': 'healthy'})
        elif self.path == '/metrics':
            self._send_response(200, self._get_metrics())
        else:
            self._send_response(404, {'error': 'Not found'})
    
    def do_POST(self):
        """Обработка POST запросов для предсказаний"""
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                prediction = self.server.ml_model.predict(data)
                self._send_response(200, {'prediction': prediction})
            except Exception as e:
                self._send_response(400, {'error': str(e)})
    
    def _send_response(self, status_code, data):
        # Отправка JSON ответа
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

class MLWebServer(HTTPServer):
    """Веб-сервер для ML моделей с кастомной логикой"""
    
    def __init__(self, host, port, ml_model):
        super().__init__((host, port), MLRequestHandler)
        self.ml_model = ml_model
        self.request_count = 0
        
    def serve_forever(self):
        print(f"🚀 ML Web Server запущен на http://{self.server_address[0]}:{self.server_address[1]}")
        super().serve_forever()