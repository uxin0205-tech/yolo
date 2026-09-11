"""唯讀模型 state 雜湊；不反序列化外部物件。"""
import hashlib

def digest(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():
        h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
