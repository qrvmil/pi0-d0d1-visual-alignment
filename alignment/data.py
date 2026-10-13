"""Seen-only deterministic input identities and augmentation replay."""
import hashlib
import numpy as np
from PIL import Image, ImageEnhance

VIEWS=('base_0_rgb','left_wrist_0_rgb')
TASKS={
    'D0':('libero_spatial','pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate'),
    'D1':('libero_spatial','pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate'),
}


def require_seen(tasks):
    if not tasks or any(t not in TASKS for t in tasks):
        raise ValueError('Only D0+D1 allowed in training/calibration/normalization')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def image_hash(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def augmentation_params(sample_id,augmentation,view):
    if view not in VIEWS or not 0<=augmentation<4:
        raise ValueError('Unknown view/augmentation')
    seed=int.from_bytes(hashlib.sha256(f'v1/0/{sample_id}/{augmentation}/{view}'.encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    side=int(rng.integers(202,225))
    return dict(seed=seed,left=int(rng.integers(0,225-side)),top=int(rng.integers(0,225-side)),
                side=side,brightness=float(rng.uniform(.9,1.1)),contrast=float(rng.uniform(.9,1.1)),
                interpolation='PIL bilinear',version=1)


def replay(image,params):
    image=np.asarray(image)
    if image.shape!=(224,224,3) or image.dtype!=np.uint8:
        raise ValueError('Replay expects224x224 uint8 RGB')
    x,y,s=params['left'],params['top'],params['side']
    img=Image.fromarray(image).crop((x,y,x+s,y+s)).resize((224,224),Image.Resampling.BILINEAR)
    img=ImageEnhance.Brightness(img).enhance(params['brightness'])
    img=ImageEnhance.Contrast(img).enhance(params['contrast'])
    return np.asarray(img).copy()


def sample_order(lengths,count,seed=0):
    """Exactly balanced task pairs; stable prefixes enable exact step-counter resume."""
    if len(lengths)!=2 or min(lengths)<=0 or count<0:raise ValueError('Two nonempty seen tasks required')
    rng=np.random.default_rng(seed)
    pools=[rng.permutation(n).tolist() for n in lengths]
    positions=[0,0];order=[]
    while len(order)<count:
        for task in rng.permutation(2):
            if positions[task]==len(pools[task]):
                pools[task]=rng.permutation(lengths[task]).tolist(); positions[task]=0
            index=pools[task][positions[task]];positions[task]+=1
            order.append((int(task),int(index),int(rng.integers(4))))
            if len(order)==count:break
    return order
