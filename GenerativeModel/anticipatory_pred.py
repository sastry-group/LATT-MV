import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from model import TransformerModel
from loaders.loader import CombinedDataset
from loaders.youtube_loader import FORMAT_RANGES, FORMAT_SIZE
from inference import generate, load_model, TOKEN_DIM, MODEL_PARAMS
from utils.analysis import *

device = torch.device("cuda")

ENS_NUMBER = 6
NUM_MODELS = 5

def generate_targets(anticipatory_window, hypers, quantiles_dict):
    random_seed = 1
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    prompt_size = -anticipatory_window
    loader = CombinedDataset("data/recons", "data/recons_lab", "data/recons_test_conformal", "data/recons_lab_test", val_split=0.05, constant_fps=True)
    models = [load_model(f'models/ensemble{ENS_NUMBER}/best_model_{i}.pth', device) for i in range(NUM_MODELS)]
    
    # Iterate through testing data
    dataset = loader.youtube_test_dataset
    mask = loader.youtube_mask
    print("Dataset size:", len(dataset))
    num_samples = len(dataset)
    res = [[] for _ in range(len(hypers))]
    for i in range(min(len(dataset), num_samples)):
        print(f"Processing Sample {i}.")
        recon_name = dataset.recons[i]
        for hit_time_idx in range(len(dataset.hit_times[i]) // 2):                
            preds = []
            for model in models:
                predicted_sequences, fps, prompt_start_idx, ground_truth = generate(
                    model, 
                    dataset, 
                    i,
                    mask, 
                    prompt_size, 
                    device,
                    1, 
                    use_mask_on_generation=True,
                    verbose=False,
                    hit_time_idx=hit_time_idx,
                    use_ground_truth=False,
                    gen_extra=10,
                    return_ground_truth_seperate=True
                )
                predicted_sequences = predicted_sequences[:, prompt_start_idx:]
                predicted_sequences = predicted_sequences * (dataset.std + 1e-8) + dataset.mean
                pred_ball = predicted_sequences[0][:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]]
                pred_ball = pred_ball.numpy()
                correct_pred(pred_ball)
                preds.append(pred_ball)
            
            preds = np.array(preds)
            mean_pred_ball = preds.mean(axis=0)
            std_pred_ball  = preds.std(axis=0)
            t = np.arange(0, len(std_pred_ball)) + 1
            
            predicted_sequences_orig, ground_truth_orig = predicted_sequences, ground_truth
            
            # Loop through hyper-param settings
            for hypers_index, params in enumerate(hypers):
                ground_truth = ground_truth_orig.detach().clone()
                predicted_sequences = predicted_sequences_orig.detach().clone()
                
                lmbd, alpha, gs, init_pos = params['lambda'], params['alpha'], params['gantry_speed'], params['initial_position']
                conformal_quantiles = quantiles_dict[alpha].T
                max_t = conformal_quantiles.shape[0]
                # Generate Confidence Regions based on Ensemble Predictions
                eps = 0.016
                n = min(max_t, len(mean_pred_ball))
                radius = conformal_quantiles[:n] * (std_pred_ball[:n] + eps*t[:n, np.newaxis])
                confidence_regions = (mean_pred_ball[:n] - radius, mean_pred_ball[:n] + radius)
                confidence_regions = np.concatenate(confidence_regions, -1).reshape(-1, 2, 3).transpose(0, 2, 1)
                confidence_regions = bound_regions(confidence_regions)
    
                # Generate Reachable Regions for the Robot arm
                reachable_regions = [generate_reachable_box(t, c=init_pos, v=[2.5, gs, gs]) for t in np.arange(0, len(confidence_regions)) / 30]
                
                contains_flgs = [contains(reachable_regions[i][2:], confidence_regions[i][2:]) and mean_pred_ball[i][0] > 1.35 for i in range(len(reachable_regions))]
                if True in contains_flgs:
                    idx = min(contains_flgs.index(True) + 1, len(contains_flgs)-1)
                    confidence_region = confidence_regions[idx]
                    target_pos = project(shrink(mean_pred_ball[idx], x0=init_pos, lmbda=lmbd), confidence_region)
                    target_pos[0] = min(max(target_pos[0], 1.35), 3.85)
                    
                    hit_time  = round(dataset.hit_times[i][2*hit_time_idx] * fps/100) - prompt_start_idx
                    
                    ground_truth = ground_truth * (dataset.std + 1e-8) + dataset.mean
                    true_ball = ground_truth[0][:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]]
                    true_ball = true_ball.numpy()
                    true_ball = true_ball[prompt_start_idx:]
                    # extension = np.vstack(extend_traj(true_ball[-4:], 14)).T
                    # true_ball = np.concatenate((true_ball, extension), 0)
                    
                    res[hypers_index].append((true_ball, hit_time, target_pos, recon_name))
                else:
                    print("Failure.")
                    ground_truth = ground_truth * (dataset.std + 1e-8) + dataset.mean
                    true_ball = ground_truth[0][:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]]
                    true_ball = true_ball.numpy()
                    true_ball = true_ball[prompt_start_idx:]
                    extension = np.vstack(extend_traj(true_ball[-4:], 14)).T
                    true_ball = np.concatenate((true_ball, extension), 0)
                    res[hypers_index].append((true_ball, hit_time, [1.8, 0, 0.9], recon_name))
                
            
    return res

conformal_quantiles_80 = np.array([
    [1.093523981478878, 1.129681134427237, 1.2057591304999202, 1.2019227733493114, 1.2074160848996367, 1.1867997514766697, 1.1211077581374984, 1.5118714267373348, 1.5406122196935563, 1.7200708756916954, 1.6672275723833259, 1.601878070549606, 1.5420266040256991, 1.5039021219154571, 1.4773247466036514, 1.4565934164049512, 1.4397710730954023, 1.4175456042788328, 1.3053528678063568, 1.2351462141038516, 1.1685176353412643, 1.1335832563493087, 1.1130492038734454, 1.116483885162204, 1.1275524752801096, 1.1358450362509325, 1.176074612349017, 1.22142820517981, 1.313832186197897, 1.4277971701642074, 1.4851179110406572, 2.061873038442578, 2.2741145779229943, 2.161448798398028, 2.5964844024667673, 2.6011802095516976], 
    [0.5357354958875526, 0.5593486184380192, 0.6410911073222672, 0.7168543842144451, 0.7979094421887539, 0.8864563150609727, 0.9433330172499761, 1.0966302777351793, 0.9342135825967516, 0.9392663213709166, 1.0191307487555858, 1.1023090253758006, 1.158392455878503, 1.2076172353824262, 1.2353194941813186, 1.219151075934287, 1.1778968533401408, 1.114906688895349, 1.0595255657299816, 1.0200178091197234, 0.9690482734448707, 0.9460373338118815, 0.9265577558169675, 0.9300664881736176, 0.9627270557786055, 0.9775245902885478, 0.998270588645502, 1.0609489233719196, 1.005681571971124, 1.0406801036218616, 1.1367757031969887, 1.270667407362337, 1.2671432854659033, 1.2556236475515792, 1.1624730602163933, 1.1605309388181868], 
    [0.21959534037517042, 0.2650302140878927, 0.3406818651984706, 0.4203143557898404, 0.5000782325800521, 0.585622588683351, 0.634013205489083, 0.7607698551063555, 0.566154567578934, 0.4900633029070012, 0.45110316897316777, 0.4433517835172919, 0.44737201518818265, 0.4565456861258582, 0.42576995523641203, 0.40554332512887503, 0.3949597209476958, 0.413014088894424, 0.40730533864717827, 0.3865298975454095, 0.3647135998490343, 0.37057865798539635, 0.37749293911708526, 0.37975452921561714, 0.4189049299576749, 0.4612925808912886, 0.5145175331214664, 0.5591338187588585, 0.6609201900860411, 0.7271812330176242, 0.7651853708832222, 0.9790708995104596, 1.0931858516496342, 1.063620806535662, 1.2532439294251363, 1.5693589365850948] 
])
conformal_quantiles_85 = np.array([
    [1.2940979910649653, 1.3578839402290792, 1.416330044232463, 1.4096028036391395, 1.4042502504792143, 1.3217428876562651, 1.3310634681022298, 1.7824959383999717, 1.8058662309996838, 1.9562780837453615, 1.8916023508336308, 1.834897063633948, 1.7894967716394483, 1.7141808042538358, 1.6995662031268857, 1.6780030251006401, 1.6569021041091196, 1.6012900431999655, 1.5027024004073093, 1.3948354362171806, 1.3068402073943644, 1.2906296825740469, 1.283539588514659, 1.2811624039052303, 1.3160637307799679, 1.329121391920035, 1.3649004062033028, 1.3816887810529044, 1.5520804173131166, 1.6013692734470046, 1.7305877448611875, 2.190120201989411, 2.5873291861618126, 2.5905082952876897, 3.006726996703031, 2.6011802095516976], 
    [0.6090266700823536, 0.6801030697286836, 0.7611157671594093, 0.8704872645847447, 0.9584607309769979, 1.020201366029445, 1.089012232767275, 1.2703387064500085, 1.0564817920717322, 1.081129801138768, 1.161052944506185, 1.251951447482963, 1.3536641902933835, 1.380738310994904, 1.3855814213917657, 1.3565391901979973, 1.3169139798644955, 1.2649551510421377, 1.161274402773064, 1.1327454981817124, 1.0880474646988092, 1.0664058815031612, 1.0454505614259901, 1.0585389953447593, 1.0722773844658784, 1.0947956221546167, 1.116176089774551, 1.17404979308419, 1.1724923575177182, 1.1910670112408641, 1.2856111384772853, 1.3353338198124292, 1.2944692757922327, 1.3129475762701481, 1.5104953437249686, 1.1605309388181868], 
    [0.2595599121472812, 0.34345163880045815, 0.4243086982864101, 0.5128322838764259, 0.5982553971761301, 0.6951305507167177, 0.7510024174505681, 0.8641113055162283, 0.6591587926587992, 0.5653114759094293, 0.5199054502555808, 0.5111241566611299, 0.51290220355998, 0.521847983143516, 0.4860720012812808, 0.4551521008070027, 0.45890456526997586, 0.471964094675431, 0.4671602428300088, 0.4526295905197145, 0.41516790982330554, 0.4298281782676325, 0.4352180167464811, 0.4390955986340088, 0.46907580625061795, 0.5185130115255348, 0.5749966985231808, 0.657121763694654, 0.7728516362842082, 0.8518052916118612, 0.9789783067219581, 1.0784054201357367, 1.343388925678266, 1.3877788834491298, 1.5078803438321868, 1.5693589365850948] 
])
conformal_quantiles_90 = np.array([
    [1.542716939945034, 1.6644300538874164, 1.6960834317077982, 1.698711576721159, 1.6664447551466994, 1.5895750129729689, 1.6219112784332343, 2.0906686851297227, 2.1267761005228696, 2.3407181852522285, 2.2727342520818024, 2.118496271776335, 2.0656047649610474, 2.0025458782104377, 1.9550444738001609, 1.967971257624644, 1.9341869374472525, 1.8737686833580423, 1.7403984175920684, 1.6189611960287562, 1.5402485166543054, 1.5200120023884045, 1.4838301242507637, 1.4997528393248802, 1.519706406481963, 1.5382522239183973, 1.6411463126250923, 1.608634882303069, 1.7736124063216139, 1.9255196026808272, 2.0961007523152566, 2.701233343511961, 3.1563250443430806, 3.007812639363857, 3.006726996703031], 
    [0.7791645751026381, 0.8414837072623009, 0.9239132769081387, 1.0504018644904736, 1.1291624733976302, 1.191320940357526, 1.2643401286733122, 1.4335230194111335, 1.207262898936403, 1.274584290683756, 1.38581296154073, 1.4506629179957695, 1.51538060675444, 1.5578661782784495, 1.5596452126221896, 1.5353440980408453, 1.4910714084366028, 1.4274887042051292, 1.3713035189083618, 1.3239224225707096, 1.2362307275618392, 1.232595413314884, 1.2096157065573738, 1.2128399334299431, 1.2084790195185675, 1.2141801970099009, 1.262848001313967, 1.2917464386007818, 1.299881263598022, 1.3451879261261497, 1.3436344424271904, 1.4154030781668894, 1.4236308385002214, 1.4809206682483584, 1.5104953437249686], 
    [0.3347325301312745, 0.44897733110035554, 0.5518706622868894, 0.6585095992325368, 0.7560512427601501, 0.83470844916534, 0.9085342290411568, 1.033516982697298, 0.7852467518012739, 0.674424539738133, 0.613490675628294, 0.5833077523845154, 0.5952865920844934, 0.5948088698052663, 0.5702765587312701, 0.541711683716907, 0.5394422829346137, 0.5547507175064065, 0.5392308585337575, 0.5397246826186486, 0.5062598373609974, 0.5122123853533951, 0.513111292367044, 0.5230218431332903, 0.5600847109259979, 0.618761202864462, 0.6822749587870801, 0.7831577916538127, 0.9247083879172162, 1.0800710827463302, 1.101028764137419, 1.2307429760891557, 1.4823916075070314, 1.436146145667105, 1.5078803438321868] 
])
quantiles_dict = {
    0.1: conformal_quantiles_90,
    0.15: conformal_quantiles_85,
    0.2: conformal_quantiles_80,
}

from itertools import product

lmbds = [0.5]
alpha = [0.1]
gantry_speed = [1.0] # [0.5, 1.0, 1.5, 2.0, 2.5]
initial_position = [
    [1.5, 0, 0.9],
    # [1.9, 0, 0.9],
    # [2.2, 0, 0.9],
]

# Generate all combinations
combinations = list(product(
    lmbds,
    alpha,
    gantry_speed,
    initial_position
))

# Convert each combination into a dictionary for clarity
for anticipatory_window in [7]: # [2, 3, 4, 5, 7]:
    combinations_dicts = [
        {
            'anticipatory_window': anticipatory_window,
            'lambda': lmbd,
            'alpha': a,
            'gantry_speed': gs,
            'initial_position': ip
        }
        for lmbd, a, gs, ip in combinations
    ]

    targets = generate_targets(anticipatory_window, combinations_dicts, quantiles_dict)

    fname = lambda d: "_".join([f"{k}_{d[k]}" for k in d]) + ".pkl"
    for d, target in zip(combinations_dicts, targets):
        file_name = f"targets/{fname(d)}"
        with open(file_name, "wb") as f:
            pickle.dump(target, f)
    