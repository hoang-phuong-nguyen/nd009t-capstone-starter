# import json
# import logging
# import sys
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# import io
# import requests
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler(sys.stdout))
# JSON_CONTENT_TYPE = 'application/json'
# JPEG_CONTENT_TYPE = 'image/jpeg'


# # Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
# def Net():
#     model = models.resnet50(pretrained=True)

#     for param in model.parameters():
#         param.requires_grad = False   

#     model.fc = nn.Sequential(
#                    nn.Linear(2048, 128),
#                    nn.ReLU(inplace=True),
#                    nn.Linear(128, 5))
#     return model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# def model_fn(model_dir):
#     print("In model_fn. Model directory is -")
#     print(model_dir)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = Net().to(device)
    
#     with open(os.path.join(model_dir, "model.pth"), "rb") as f:
#         checkpoint = torch.load(f, map_location=device)
#         model.load_state_dict(checkpoint)
#         logger.info('model loaded successfully')
#     model.eval()
#     return model




# def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
#     logger.info('Deserializing the input data.')
#     # process an image uploaded to the endpoint
#     #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
#     logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
#     logger.debug(f'Request body TYPE is: {type(request_body)}')
#     if content_type == JPEG_CONTENT_TYPE: 
#         return Image.open(io.BytesIO(request_body))
#     logger.debug('SO loded JPEG content')
#     # process a URL submitted to the endpoint
    
#     if content_type == JSON_CONTENT_TYPE:
#         #img_request = requests.get(url)
#         logger.debug(f'Request body is: {request_body}')
#         request = json.loads(request_body)
#         logger.debug(f'Loaded JSON object: {request}')
#         url = request['url']
#         img_content = requests.get(url).content
#         return Image.open(io.BytesIO(img_content))
    
#     raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# # inference
# def predict_fn(input_object, model):
#     logger.info('In predict fn')
#     test_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])
#     logger.info("transforming input")
#     input_object=test_transform(input_object)
    
#     with torch.no_grad():
#         logger.info("Calling model")
#         prediction = model(input_object.unsqueeze(0))
#     return prediction

# # postprocess
# def output_fn(predictions, content_type):
#     assert content_type == "application/json"
#     res = predictions.cpu().numpy().tolist()
#     return json.dumps(res)

# '''
# # Serialize the prediction result into the desired response content type
# def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
#     logger.info('Serializing the generated output.')
#     if accept == JSON_CONTENT_TYPE: 
#         logger.debug(f'Returning response {json.dumps(prediction)}')
#         return json.dumps(prediction), accept
#     raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))
# '''

# Based on tutorial from: https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html
import json
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

def create_pretrained_model():
    '''
    Create pretrained resnet50 model
    When creating our model we need to freeze all the convolutional layers which we do by their requires_grad() attribute to False. 
    We also need to add a fully connected layer on top of it which we do use the Sequential API.
    '''
    model = models.resnet50(pretrained=True, progress=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 5))
    return model

'''
Load model
'''  
def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    '''
    Initialize pretrained model
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model=create_pretrained_model()
    model.to(device)
    
    '''
    Load model state from directory
    '''    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the dog-classifier model")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
        print('Model loaded successfully')
    model.eval()
    return model


'''
This function handles data decoding. 
'''  
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    print('Deserializing the input data.')
    print(f'Request body CONTENT-TYPE is: {content_type}')
    print(f'Request body TYPE is: {type(request_body)}')
    
    if content_type == JPEG_CONTENT_TYPE: 
        print('Loaded JPEG content')
        return Image.open(io.BytesIO(request_body))
    
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        print('Loaded JSON content')
        print(f'Request body is: {request_body}')
        request = json.loads(request_body)
        print(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

'''
Run inference
''' 
def predict_fn(input_object, model):
    print('In predict fn')
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    
    print("Transforming input")
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        print("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction

'''
Data post-process
''' 
def output_fn(predictions, content_type):
    print(f'Postprocess CONTENT-TYPE is: {content_type}')
    assert content_type == JSON_CONTENT_TYPE
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)