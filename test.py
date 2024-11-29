import os
from fid.model import load_fidnet_v3
from util.metric import compute_generative_model_scores, compute_maximum_iou, compute_overlap, compute_alignment
import pickle as pk
from tqdm import tqdm
from util.datasets.load_data import init_dataset
from util.visualization import save_image,save_label,save_label_with_size
from util.constraint import *
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
from util.seq_util import sparse_to_dense, loader_to_list, pad_until
import argparse
from onehot.diffusiononehot import *
from model.onehotganZ import Generator,Generator_rico25,Generator_rico25_c_sp
from model.layoutganpp import Generator as layoutganpp
import time
from collections import Counter

def test_fid_feat(dataset_name, device='cuda', batch_size=20):

    if os.path.exists(f'./fid/feature/fid_feat_test_{dataset_name}.pk'):
        feats_test = pk.load(open(f'./fid/feature/fid_feat_test_{dataset_name}.pk', 'rb'))
        return feats_test

    # prepare dataset
    main_dataset, main_dataloader = init_dataset(dataset_name, './datasets', batch_size=batch_size,
                                                 split='test', shuffle=False, transform=None)

    fid_model = load_fidnet_v3(main_dataset, './fid/FIDNetV3', device=device)
    feats_test = []

    with tqdm(enumerate(main_dataloader), total=len(main_dataloader), desc=f'Get feature for FID',
              ncols=200) as pbar:

        for i, data in pbar:

            bbox, label, _, mask = sparse_to_dense(data)
            label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)
            padding_mask = ~mask

            with torch.set_grad_enabled(False):
                feat = fid_model.extract_features(bbox, label, padding_mask)
            feats_test.append(feat.detach().cpu())

    pk.dump(feats_test, open(f'./fid/feature/fid_feat_test_{dataset_name}.pk', 'wb'))

    return feats_test


def test_layout_uncond(G,batch_size=128, dataset_name='publaynet', seq_dim = 10, test_plot=False,
                       save_dir='./plot/test', beautify=False):

    G.eval()
    device = 'cuda:0'
    n_batch_dict = {'publaynet': int(44 * 256 / batch_size), 'rico13': int(17 * 256 / batch_size),
                    'rico25': int(17 * 256 / batch_size), 'magazine': int(512 / batch_size),
                    'crello': int(2560 / batch_size)}
    n_batch = n_batch_dict[dataset_name]

    # prepare dataset
    main_dataset, _ = init_dataset(dataset_name, './datasets', batch_size=batch_size, split='test')

    fid_model = load_fidnet_v3(main_dataset, './fid/FIDNetV3', device=device)
    feats_test = test_fid_feat(dataset_name, device=device, batch_size=20)
    feats_generate = []

    align_sum = 0
    overlap_sum = 0
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    with torch.no_grad():
        for i in tqdm(range(n_batch), desc='uncond testing', ncols=200, total=n_batch):
            x_t_1 = torch.randn(batch_size, 25, seq_dim, device=device)
            z = torch.randn(batch_size, 25, args.latent_dim,device=device)
            mask = torch.ones_like(x_t_1,dtype=torch.bool)
            layout_0 = uncond_sample_from_model(pos_coeff, G, args.num_timesteps, x_t_1, mask,args, latent_z= z)
            bbox_generated, label, mask = finalize(layout_0,num_class=args.num_label) 
            if beautify and dataset_name=='publaynet':
                bbox_generated, mask = post_process(bbox_generated, mask, w_o=1)
            elif beautify and (dataset_name=='rico25' or dataset_name=='rico13'):
                bbox_generated, mask = post_process(bbox_generated, mask, w_o=0)
            padding_mask = ~ mask
            label[mask == False] = 0
            if torch.isnan(bbox_generated[0, 0, 0]):
                print('not a number error')
                return None

            # accumulate align and overlap
            align_norm = compute_alignment(bbox_generated, mask)
            align_sum += torch.mean(align_norm)
            overlap_score = compute_overlap(bbox_generated, mask)
            overlap_sum += torch.mean(overlap_score)


            with torch.set_grad_enabled(False):
                feat = fid_model.extract_features(bbox_generated, label, padding_mask)
            feats_generate.append(feat.cpu())

            if test_plot and i <= 10:
                cond = 'uncond'
                img = save_image(bbox_generated[:20], label[:20], mask[:20], draw_label=False, dataset=dataset_name)
                plt.figure(figsize=[12, 12])
                plt.imshow(img)
                plt.tight_layout()
                plt.savefig(f'./plot/test/cond_{cond}_{dataset_name}_{i}.png')
                # plt.close()
    
    result = compute_generative_model_scores(feats_test, feats_generate)
    fid = result['fid']

    align_final = 100 * align_sum / n_batch
    overlap_final = 100 * overlap_sum / n_batch

    print(f'uncond, align: {align_final}, fid: {fid}, overlap: {overlap_final}')

    return align_final, fid, overlap_final

def ganpp_test_layout_cond(model, batch_size=256, cond='c', dataset_name='publaynet', seq_dim=10,
                     test_plot=False, save_dir='./plot/test', beautify=False):

    assert cond in {'c', 'cwh', 'complete'}
    model.eval()
    device = 'cuda:0'

    # prepare dataset
    main_dataset, main_dataloader = init_dataset(dataset_name, './datasets', batch_size=batch_size,
                                                 split='test', shuffle=False, transform=None)

    layouts_main = loader_to_list(main_dataloader)
    layout_generated = []

    fid_model = load_fidnet_v3(main_dataset, './fid/FIDNetV3', device=device)
    feats_test = test_fid_feat(dataset_name, device=device, batch_size=20)
    feats_generate = []
    time_total = 0
    align_sum = 0
    overlap_sum = 0
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    with torch.no_grad():
        with tqdm(enumerate(main_dataloader), total=len(main_dataloader), desc=f'cond: {cond} generation',disable=False,
                ncols=200) as pbar:

            for i, data in pbar:

                bbox, label, _, mask = sparse_to_dense(data)
                label, bbox, mask = pad_until(label, bbox, mask, max_seq_length=25)
                label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)

                z = torch.randn(label.size(0), label.size(1),
                            args.latent_dim, device=device)
                if cond =='cwh':
                    z[:,:,2:]=bbox[:,:,2:]
                padding_mask = ~mask
                time_start = time.time()
                bbox_generated = model(z, label, padding_mask)
                time_end = time.time()
                time_total += time_end - time_start
                if cond == 'cwh':
                    bbox_generated[:,:,2:]=bbox[:,:,2:]
                if test_plot and i <= 10:
                    img = save_image(bbox_generated[:20], label[:20], mask[:20],
                                    draw_label=False, dataset=dataset_name)
                    plt.figure(figsize=[12, 12])
                    plt.imshow(img)
                    plt.tight_layout()
                    plt.savefig(f'./plot/layoutgan++/cond_{cond}_{dataset_name}_{i}.png')
                    plt.close()
               
    time_per_sample = 1e3*(time_total)/(len(main_dataset)*100)              
    print(time_per_sample)
    


def test_layout_cond(model, batch_size=256, cond='c', dataset_name='publaynet', seq_dim=10,
                     test_plot=False, save_dir='./plot/test', beautify=False):

    assert cond in {'c', 'cwh', 'complete'}
    model.eval()
    device = 'cuda:0'

    # prepare dataset
    main_dataset, main_dataloader = init_dataset(dataset_name, './datasets', batch_size=batch_size,
                                                 split='test', shuffle=False, transform=None)

    layouts_main = loader_to_list(main_dataloader)
    layout_generated = []

    fid_model = load_fidnet_v3(main_dataset, './fid/FIDNetV3', device=device)
    feats_test = test_fid_feat(dataset_name, device=device, batch_size=20)
    feats_generate = []

    align_sum = 0
    overlap_sum = 0
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    publaynet_labels = [
        "text",
        "title",
        "list",
        "table",
        "figure",
    ]
    rico25_labels = [
        "Text",
        "Image",
        "Icon",
        "Text Button",
        "List Item",
        "Input",
        "Background Image",
        "Card",
        "Web View",
        "Radio Button",
        "Drawer",
        "Checkbox",
        "Advertisement",
        "Modal",
        "Pager Indicator",
        "Slider",
        "On/Off Switch",
        "Button Bar",
        "Toolbar",
        "Number Stepper",
        "Multi-Tab",
        "Date Picker",
        "Map View",
        "Video",
        "Bottom Navigation",
    ]
    with torch.no_grad():

        with tqdm(enumerate(main_dataloader), total=len(main_dataloader), desc=f'cond: {cond} generation',
                  ncols=200) as pbar:

            for i, data in pbar:

                bbox, label, _, mask = sparse_to_dense(data)
                label, bbox, mask = pad_until(label, bbox, mask, max_seq_length=25)
                label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)

                label[mask == False] = seq_dim - 5

                label_oh = torch.nn.functional.one_hot(label, num_classes=args.num_label).to(torch.float32)       
                layout_input = torch.cat((label_oh, bbox), dim=2).to(device)

                l_0,fix_mask = sample_diffusion(pos_coeff,model, layout_input, mask, cond, args)
                bbox_generated, label_generated, mask_generated = finalize(l_0, num_class=args.num_label)

                if beautify and dataset_name == 'publaynet':
                    bbox_generated, mask_generated = post_process(bbox_generated, mask_generated, w_o=1)
                elif beautify and (dataset_name == 'rico25' or dataset_name == 'rico13'):
                    bbox_generated, mask_generated = post_process(bbox_generated, mask_generated, w_o=0)

                padding_mask = ~ mask_generated

                # accumulate align and overlap
                align_norm = compute_alignment(bbox_generated, mask)
                align_sum += torch.mean(align_norm)
                overlap_score = compute_overlap(bbox_generated, mask)
                overlap_sum += torch.mean(overlap_score)

                label_generated[label_generated == seq_dim - 5] = 0
                for j in range(bbox.shape[0]):
                    mask_single = mask_generated[j, :]
                    bbox_single = bbox_generated[j, mask_single, :]
                    label_single = label_generated[j, mask_single]

                    layout_generated.append((bbox_single.to('cpu').numpy(), label_single.to('cpu').numpy()))

                # record for FID
                with torch.set_grad_enabled(False):
                    feat = fid_model.extract_features(bbox_generated, label_generated, padding_mask)
                feats_generate.append(feat.cpu())

                if test_plot and i <= 10:
                    # if dataset_name == 'publaynet':
                    #     labels = [
                    #             [publaynet_labels[index] for col_idx, index in enumerate(row) if (mask_generated[:20])[row_idx][col_idx]]
                    #             for row_idx, row in enumerate(label_generated[:20])
                    #     ]
                    #     simplified_labels = []
                    #     for row in labels:
                    #         counter = Counter(row)
                    #         formatted_row = [f"{label}*{count}" if count > 1 else label for label, count in counter.items()]
                    #         simplified_labels.append(formatted_row)
                    #     filename = 'label_publaynet.txt'
                    # else :
                    #     labels = [
                    #             [rico25_labels[index] for col_idx, index in enumerate(row) if (mask_generated[:20])[row_idx][col_idx]]
                    #             for row_idx, row in enumerate(label_generated[:20])
                    #     ]
                    #     simplified_labels = []
                    #     for row in labels:
                    #         counter = Counter(row)
                    #         formatted_row = [f"{label}*{count}" if count > 1 else label for label, count in counter.items()]
                    #         simplified_labels.append(formatted_row)
                    #     filename = 'label_rico25.txt'

                    # with open(filename,'a') as file:
                    #     for item in simplified_labels:
                    #         file.write(str(item)+'\n')   
                    img = save_image(bbox_generated[:20], label_generated[:20], mask_generated[:20], draw_label=False, dataset=dataset_name)
                    plt.figure(figsize=[12, 12])
                    plt.imshow(img)
                    plt.tight_layout()
                    plt.savefig(f'./plot/test/save_label_{dataset_name}_{i}.png')
                    plt.close()


                    # img = save_image(bbox[:20], label[:20], fix_mask[:20], draw_label=False, dataset=dataset_name)
                    # plt.figure(figsize=[12, 12])
                    # plt.imshow(img)
                    # plt.tight_layout()
                    # plt.savefig(f'./plot/test/condition_complete_{dataset_name}_{i}.png')
                    # plt.close()
                    # img = save_image(bbox[:20], label[:20], mask[:20], draw_label=False, dataset=dataset_name)
                    # plt.figure(figsize=[12, 12])
                    # plt.imshow(img)
                    # plt.tight_layout()
                    # plt.savefig(os.path.join(save_dir, f'{dataset_name}_real_{i}.png'))
                    # plt.close()

    maxiou = compute_maximum_iou(layouts_main, layout_generated)
    result = compute_generative_model_scores(feats_test, feats_generate)
    fid = result['fid']

    align_final = 100 * align_sum / len(main_dataloader)
    overlap_final = 100 * overlap_sum / len(main_dataloader)

    print(f'cond {cond}, align: {align_final}, fid: {fid}, maxiou: {maxiou}, overlap: {overlap_final}')

    return align_final, fid, maxiou, overlap_final


def test_layout_refine(model, batch_size=256,cond='refine', dataset_name='publaynet', seq_dim=10,
                     test_plot=False, save_dir='./plot/test', beautify=False):

    model.eval()
    device = 'cuda:0'
    n_batch_dict = {'publaynet': 44, 'rico13': 17, 'rico25': 17, 'magazine': 2, 'crello': 10}
    n_batch = n_batch_dict[dataset_name]

    # prepare dataset
    main_dataset, main_dataloader = init_dataset(dataset_name, './datasets', batch_size=batch_size,
                                                 split='test', shuffle=False, transform=None)

    layouts_main = loader_to_list(main_dataloader)
    layout_generated = []

    fid_model = load_fidnet_v3(main_dataset, './fid/FIDNetV3', device=device)
    feats_test = test_fid_feat(dataset_name, device=device, batch_size=20)
    feats_generate = []

    align_sum = 0
    overlap_sum = 0
    pos_coeff = Posterior_Coefficients(args, device)
    coeff = Diffusion_Coefficients(args, device)
    with torch.no_grad():

        with tqdm(enumerate(main_dataloader), total=min(n_batch, len(main_dataloader)), desc=f'refine generation',
                  ncols=200) as pbar:

            for i, data in pbar:
                if i == min(n_batch, len(main_dataloader)):
                    break

                bbox, label, _, mask = sparse_to_dense(data)
                label, bbox, mask = pad_until(label, bbox, mask, max_seq_length=25)

                label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)

                # shift to center
                bbox_in_noisy = torch.clamp(bbox + 0.1 * torch.randn_like(bbox), min=0, max=1).to(device)
                #
                # set mask to label 5
                label[mask == False] = seq_dim - 5

                label_oh = torch.nn.functional.one_hot(label, num_classes=seq_dim - 4).to(torch.float32).to(device)
                layout_input = torch.cat((label_oh, bbox), dim=2).to(device)
                l_0 = refine_sample_diffusion(coeff,model, layout_input, mask, cond, args)
                bbox_refined, label_generated, mask_generated = finalize(l_0, num_class=args.num_label)
                if beautify and dataset_name=='publaynet':
                    bbox_refined, mask = post_process(bbox_refined, mask, w_o=1)
                elif beautify and (dataset_name=='rico25' or dataset_name=='rico13'):
                    bbox_refined, mask = post_process(bbox_refined, mask, w_o=0)
                padding_mask = ~mask

                # accumulate align and overlap
                align_norm = compute_alignment(bbox_refined, mask)
                align_sum += torch.mean(align_norm)
                overlap_score = compute_overlap(bbox_refined, mask)
                overlap_sum += torch.mean(overlap_score)

                # record for max_iou
                label[label == seq_dim - 5] = 0

                for j in range(bbox_refined.shape[0]):
                    mask_single = mask[j, :]
                    bbox_single = bbox_refined[j, mask_single, :]
                    label_single = label[j, mask_single]

                    layout_generated.append((bbox_single.to('cpu').numpy(), label_single.to('cpu').numpy()))

                # record for FID
                with torch.set_grad_enabled(False):
                    feat = fid_model.extract_features(bbox_refined, label, padding_mask)
                feats_generate.append(feat.cpu())


                if test_plot and i <= 10:
                    img = save_image(bbox_refined[:9], label[:9], mask[:9],
                                     draw_label=False, dataset=dataset_name)
                    plt.figure(figsize=[12, 12])
                    plt.imshow(img)
                    plt.tight_layout()
                    plt.savefig(f'./plot/test/refine_{dataset_name}_{i}.png')
                    plt.close()

                    img = save_image(bbox[:9], label[:9], mask[:9], draw_label=False, dataset=dataset_name)
                    plt.figure(figsize=[12, 12])
                    plt.imshow(img)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'{dataset_name}_real.png'))
                    plt.close()

    maxiou = compute_maximum_iou(layouts_main, layout_generated)
    result = compute_generative_model_scores(feats_test, feats_generate)
    fid = result['fid']

    align_final = 100 * align_sum / len(main_dataloader)
    overlap_final = 100 * overlap_sum / len(main_dataloader)

    print(f'refine, align: {align_final}, fid: {fid}, maxiou: {maxiou}, overlap: {overlap_final}')
    return align_final, fid, maxiou, overlap_final

def test_layout_speed(model, batch_size=256, cond='c', dataset_name='publaynet', seq_dim=10,
                     test_plot=False, save_dir='./plot/test'):

    model.eval()
    device = 'cuda:0'

    # prepare dataset
    main_dataset, main_dataloader = init_dataset(dataset_name, './datasets', batch_size=batch_size,
                                                 split='test', shuffle=False, transform=None)

    layouts_main = loader_to_list(main_dataloader)
    data_prepare = torch.randn(batch_size,25,seq_dim,device=device)
    z_prepare = torch.rand(batch_size,25,args.latent_dim,device=device)
    mask = torch.zeros(batch_size,25,device=device)
    time_total = 0
    with torch.no_grad():
        model(data_prepare[:,:,:num_class],data_prepare[:,:,num_class:],mask,z_prepare)
    
    pos_coeff = Posterior_Coefficients(args, device)
    with torch.no_grad():
        for i in range(1):
            with tqdm(enumerate(main_dataloader), total=len(main_dataloader), desc=f'cond: {cond} generation',disable=True,
                    ncols=200) as pbar:

                for i, data in pbar:

                    bbox, label, _, mask = sparse_to_dense(data)
                    label, bbox, mask = pad_until(label, bbox, mask, max_seq_length=25)
                    label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)

                    label[mask == False] = seq_dim - 5

                    label_oh = torch.nn.functional.one_hot(label, num_classes=args.num_label).to(torch.float32)       
                    layout_input = torch.cat((label_oh, bbox), dim=2).to(device)
                    time_start = time.time()
                    l_0 = speed_sample_diffusion(pos_coeff,model, layout_input, mask, cond, args)
                    time_end = time.time()
                    time_total += time_end - time_start
                    bbox_generated, label_generated, mask_generated = finalize(l_0, num_class=args.num_label)
                    label_generated[label_generated == seq_dim - 5] = 0
    time_per_sample = 1e3*(time_total)/(len(main_dataset)*1)              
    print(time_per_sample)
    print(len(main_dataset))
    return time_per_sample    

    

def test_all(model, dataset_name='publaynet', seq_dim=10, test_plot=False, save_dir='./plot/test', batch_size=256,
             beautify=False):

    align_uncond, fid_uncond, overlap_uncond = test_layout_uncond(model, batch_size=batch_size, dataset_name=dataset_name,
                                                  test_plot=test_plot, save_dir=save_dir, beautify=beautify)
    align_c, fid_c, maxiou_c, overlap_c = test_layout_cond(model, batch_size=batch_size, cond='c',
                                                dataset_name=dataset_name, seq_dim=seq_dim,
                                                test_plot=test_plot, save_dir=save_dir, beautify=beautify)
    align_cwh, fid_cwh, maxiou_cwh, overlap_cwh = test_layout_cond(model, batch_size=batch_size, cond='cwh',
                                                      dataset_name=dataset_name, seq_dim=seq_dim,
                                                      test_plot=test_plot, save_dir=save_dir, beautify=beautify)
    align_complete, fid_complete, maxiou_complete, overlap_complete = test_layout_cond(model, batch_size=batch_size,
                                                                     cond='complete', dataset_name=dataset_name,
                                                                     seq_dim=seq_dim, test_plot=test_plot,
                                                                     save_dir=save_dir, beautify=beautify)
    align_r, fid_r, maxiou_r, overlap_r = test_layout_refine(model, batch_size=batch_size,
                                            dataset_name=dataset_name, seq_dim=seq_dim,
                                            test_plot=test_plot, save_dir=save_dir, beautify=beautify)

    # fid_total = fid_uncond + fid_c + fid_cwh + fid_complete
    # print(f'total fid: {fid_total}')
    # return fid_total


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, help="batch_size", type=int)
    parser.add_argument("--device", default='cuda:0', help="which GPU to use", type=str)
    parser.add_argument("--dataset", default='rico25',
                        help="choose from [publaynet, rico13, rico25]", type=str)
    parser.add_argument("--data_dir", default='./datasets', help="dir of datasets", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--feature_dim", default=2048, help="feature_dim", type=int)
    parser.add_argument("--dim_transformer", default=1024, help="dim_transformer", type=int)
    parser.add_argument("--nhead", default=16, help="nhead attention", type=int)
    parser.add_argument("--nlayer", default=4, help="nlayer", type=int)
    parser.add_argument("--experiment", default='c', help="experiment setting [uncond, c, cwh, complete, speed, all]", type=str)
    parser.add_argument('--plot', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--beautify', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_save_dir", default='./plot/test', help="dir to save generated plot of layouts", type=str)
    parser.add_argument('--D_d_model', type=int, default=256,
                        help='d_model for discriminator')
    parser.add_argument('--D_nhead', type=int, default=4,
                        help='nhead for discriminator')
    parser.add_argument('--D_num_layers', type=int, default=8,
                        help='num_layers for discriminator')
    parser.add_argument('--G_d_model', type=int, default=256,
                        help='d_model for generator')
    parser.add_argument('--G_nhead', type=int, default=4,
                        help='nhead for generator')
    parser.add_argument('--G_num_layers', type=int, default=8,
                        help='num_layers for generator')
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument('--beta_min', type=float, default= 0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--b_size', type=int, default=4,
                        help='latent size')
    parser.add_argument('--num_label', type=int, default=4,)
    parser.add_argument('--out_dim',type=int,default=32)
    args = parser.parse_args()
    torch.manual_seed(3486)
    # prepare data
    train_dataset, train_loader = init_dataset(args.dataset, args.data_dir, batch_size=args.batch_size,
                                               split='train', shuffle=True)
    num_class = train_dataset.num_classes + 1
    args.num_label = num_class
    # G = layoutganpp(args.latent_dim, num_class-1,
    #                  d_model=args.G_d_model,
    #                  nhead=args.G_nhead,
    #                  num_layers=args.G_num_layers,
    #                  ).to('cuda:0')   

    #set up model
    if args.dataset == 'publaynet' and args.experiment in ['cwh']:#Generator for uncond
        G = Generator_rico25_c_sp(args.b_size,args.num_label,
            d_model=args.G_d_model,
            nhead=args.G_nhead,
            num_layers=args.G_num_layers,
            z_dim = args.latent_dim
            ).to('cuda:0')   
    elif args.dataset == 'publaynet' and args.experiment in ['c','refine']:#Generator for uncond
        G = Generator_rico25(args.b_size,args.num_label,
            d_model=args.G_d_model,
            nhead=args.G_nhead,
            num_layers=args.G_num_layers,
            z_dim = args.latent_dim
            ).to('cuda:0')       
           
    elif args.dataset == 'publaynet' and args.experiment in ['uncond','complete']:#Generator for uncond
        G = Generator_rico25(args.b_size,args.num_label,
            d_model=args.G_d_model,
            nhead=args.G_nhead,
            num_layers=args.G_num_layers,
            z_dim = args.latent_dim
            ).to('cuda:0')       
    
    elif args.dataset == 'rico25' and args.experiment in ['c','cwh','refine']:
        G = Generator_rico25_c_sp(args.b_size,args.num_label,
                    d_model=args.G_d_model,
                    nhead=args.G_nhead,
                    num_layers=args.G_num_layers,
                    z_dim = args.latent_dim,
                    out_dim= args.out_dim
                    ).to('cuda:0')   
    elif args.dataset == 'rico25' and args.experiment in ['uncond','complete']:
        G = Generator_rico25(args.b_size,args.num_label,
                    d_model=args.G_d_model,
                    nhead=args.G_nhead,
                    num_layers=args.G_num_layers,
                    z_dim = args.latent_dim
                    ).to('cuda:0')       
    
    
    #ablation study on timestep
    #ckpt = torch.load('/home/ganzhaoxing/diffusion_layout_gan/output/publaynet/LayoutGAN++/20240502005324772042/model_best.pth.tar', map_location='cuda:0')
    ckpt = torch.load('/home/ganzhaoxing/diffusion_layout_gan/output/publaynet/LayoutGAN++/20240502005312401924/model_best.pth.tar', map_location='cuda:0')
    #ckpt = torch.load('/home/ganzhaoxing/diffusion_layout_gan/output/publaynet/LayoutGAN++/20240502005302249467/model_best.pth.tar', map_location='cuda:0')
    #ckpt = torch.load('/home/ganzhaoxing/diffusion_layout_gan/output/publaynet/LayoutGAN++/20240414202238777300/model_best.pth.tar', map_location='cuda:0')
    G.load_state_dict(ckpt['netG'])
    G.eval()
    # if args.experiment in ['c', 'cwh']:
    #     ganpp_test_layout_cond(G, batch_size=args.batch_size, cond=args.experiment,
    #                                           dataset_name=args.dataset, seq_dim=num_class + 4,
    #                                           test_plot=args.plot, save_dir=args.plot_save_dir, beautify=args.beautify)
    if args.experiment == 'uncond':
        test_layout_uncond(G, batch_size=args.batch_size,
                                                      dataset_name=args.dataset, test_plot=args.plot,seq_dim=num_class + 4,
                                                      save_dir=args.plot_save_dir, beautify=args.beautify)
    elif args.experiment in ['c', 'cwh', 'complete']:
        test_layout_cond(G, batch_size=args.batch_size, cond=args.experiment,
                                              dataset_name=args.dataset, seq_dim=num_class + 4,
                                              test_plot=args.plot, save_dir=args.plot_save_dir, beautify=args.beautify)
    elif args.experiment == 'refine':
        test_layout_refine(G, batch_size=args.batch_size,cond=args.experiment,
                                                dataset_name=args.dataset, seq_dim=num_class + 4,
                                                test_plot=args.plot, save_dir=args.plot_save_dir, beautify=args.beautify)
    elif args.experiment == 'speed':
        test_layout_speed(G, batch_size=args.batch_size,
                                                dataset_name=args.dataset, seq_dim=num_class + 4,
                                                test_plot=args.plot, save_dir=args.plot_save_dir)

    elif args.experiment == 'all':
        test_all(G, dataset_name=args.dataset, seq_dim=num_class + 4, test_plot=args.plot,
                 save_dir=args.plot_save_dir, batch_size=args.batch_size, beautify=args.beautify)
    else:
        raise Exception('experiment setting undefined')




