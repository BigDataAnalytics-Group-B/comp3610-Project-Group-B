from flask import Flask
from flask_uploads import DOCUMENTS, UploadSet, configure_uploads
from flask_cors import CORS

from App.database import init_db
from App.config import config

def configure_app(app, config, overrides):
    for key, value in config.items():
        if key in overrides:
            app.config[key] = overrides[key]
        else:
            app.config[key] = config[key]

def create_app(config_overrides={}):
    app = Flask(__name__, static_url_path='/static')
    configure_app(app, config, config_overrides)
    app.config['UPLOADED_PHOTOS_DEST'] = "App/uploads"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['PREFERRED_URL_SCHEME'] = 'https'
    CORS(app)
    photos = UploadSet('photos', DOCUMENTS)
    configure_uploads(app, photos)
    init_db(app)
    return app
