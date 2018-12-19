import requests
import json
import shutil
from time import sleep
import re
import mlcode

token = "742399582:AAHY_IvizrRsmjAnLkv_bCcxtm8kHbaYl2g"
baseurl = "https://api.telegram.org/bot{}/".format(token)

def get_updates(offset = None):
    while True:
        url = baseurl + 'getUpdates'
        if offset:
            url += '?offset={}'.format(offset) 
        res = requests.get(url)
        while (res.status_code !=200 or len(res.json()['result'])== 0):
            sleep(1)
            res = requests.get(url)
        print(res.url)
        return res.json()
    
def last_update_id(res):
    return res['result'][-1]['update_id']

def get_start_chats(updates):
    start_chats = []
    for i in updates['result']:
        if 'text' in i['message'] and re.search("start",i['message']['text'].lower()):
            start_chats.append(i['message']['chat']['id'])
            print("Starting chat with", i['message']['chat']['username'])
    return start_chats

def get_photo_chats(updates):
    photo_chats = []
    for i in updates['result']:
        if 'photo' in i['message']:
            photo_chats.append((i['message']['chat']['id'],
                               i['message']['photo'][-1]['file_id']))
            print(i['message']['chat']['username'], "sent an image")
    return photo_chats

def send_start_msg(chats):
    url = baseurl + 'sendMessage'
    text = """Hey, Welcome to the Image Captioning Center!
I am PyCaption Bot and I will help you decide a caption for your images.
Send me an image and I will suggest a caption for it"""
    for cid in chats:
        msg = requests.get(url, params={'chat_id':cid, 'text':text})
        while msg.status_code!=200:
            sleep(1)
            msg = requests.get(url, params={'chat_id':cid, 'text':text})
    return 'ok'

def get_save_file(file_id):
    fileurl = baseurl + 'getFile'
    dwnurl = 'https://api.telegram.org/file/bot{}/'.format(token)
    
    fl = requests.get(fileurl, params={'file_id':file_id})
    path = fl.json()['result']['file_path']
    dwn = requests.get(dwnurl+"{}".format(path))
    while dwn.status_code!=200:
        sleep(1)
        fl = requests.get(fileurl, params={'file_id':file_id})
        path = fl.json()['result']['file_path']
        dwn = requests.get(dwnurl+"{}".format(path))
        
    with open("./images/"+file_id+".jpg", 'wb') as img:
            for chunk in dwn: img.write(chunk)
                
def send_photo_reply(photo_chats):
    sendurl = baseurl + 'sendMessage'
    for (chat_id, file_id) in photo_chats:
        msg = "Image recieved!\nThinking of a caption..."
        send = requests.get(sendurl, params={'chat_id':chat_id, 'text':msg})
        while send.status_code!=200:
            send = requests.get(sendurl, params={'chat_id':chat_id, 'text':msg})
        get_save_file(file_id)
    
    for (chat_id, file_id) in photo_chats:
        fname = "./images/{}.jpg".format(file_id)
        caption = mlcode.apply_model_to_image(fname).capitalize()
        send = requests.get(sendurl, params={'chat_id':chat_id, 
                                        'text':"Here's the best caption I can come up with"})
        while send.status_code!=200:
            send = requests.get(sendurl, params={'chat_id':chat_id, 
                                        'text':"Here's the best caption I can come up with"})
        
        send = requests.get(sendurl, params={'chat_id':chat_id, 
                                        'text':'"""\n{}\n"""'.format(caption)})
        while send.status_code!=200:
            send = requests.get(sendurl, params={'chat_id':chat_id, 
                                        'text':'"""\n{}\n"""'.format(caption)})
    return 'ok'

def run():
    offset = None
    while True:
        try:
            updates = get_updates(offset)
            offset = last_update_id(updates)+1
            start_chats = get_start_chats(updates)
            send_start_msg(start_chats)
            photo_chats = get_photo_chats(updates)
            send_photo_reply(photo_chats)
        except KeyboardInterrupt:
            break
            
run()