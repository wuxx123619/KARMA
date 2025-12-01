import json
import requests
qid2ent ={}

def search_qid(qid):
    try:
        params = {
            'ids': qid,  # 实体id,可多个，比如'Q123|Q456'
            'action': 'wbgetentities',
            'format': 'json',
            'languages': 'en',
        }

        # 访问
        get = requests.get(url="https://www.wikidata.org/w/api.php", params=params, timeout=5, verify=False)
    except:
        get = search_qid(qid)
    return get

def get_qid(qid):
    get = search_qid(qid)
    try:
        re_json = get.json()
    except json.JSONDecodeError as e:
        re_json = {}
    return re_json

def qid2ent_extract(filename):
    with open(filename) as f:
        data = json.load(f)
    for Qid in data['concepts']:
        ent_name = data['concepts'][Qid]
        qid2ent[Qid] = ent_name['name']
    for line in data['entities']:
        ent_name = data['entities'][line]['name']
        if line in qid2ent.keys():
            pass
        else:
            qid2ent[line] = ent_name
    for line in data['entities']:
        evi_data = data['entities'][line]
        for qdata in evi_data['relations']:
            qid = qdata['object']
            if qid in qid2ent.keys():
                pass
            else:
                qid_data = get_qid(qid)
                if qid_data['entities'][qid]['labels'] == {}:
                    tg_ent = qid_data['entities'][qid]['descriptions']['en']['value']
                else:
                    tg_ent = qid_data['entities'][qid]['labels']['en']['value']
                qid2ent[qid] = tg_ent


if __name__ == '__main__':
    qid2ent_extract("kb.json")
    with open("qid2ent.json", 'w', encoding='utf-8') as file:
        json.dump(qid2ent, file)
