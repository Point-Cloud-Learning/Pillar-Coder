import torch
from torch import nn
import torch.nn.functional as F
from PillarCoder.model.Tool import query_ball_point, index_points


class Semantic_Embedding(nn.Module):
    def __init__(self, sem_emb_type, init_dim, dim, radius, query_num):
        super(Semantic_Embedding, self).__init__()
        self.sem_emb_type = sem_emb_type
        self.radius = radius
        self.query_num = query_num
        self.trans = nn.Sequential(
            nn.Linear(init_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True)
        )
        if self.sem_emb_type == "complex_sem":
            self.sem_trans = nn.Sequential(
                nn.Linear(dim * 2, int(dim / 2)),
                nn.LayerNorm(int(dim / 2)),
                nn.ReLU(inplace=True)
            )

    def forward(self, features, coordinates, dis_mats):
        x = self.trans(features)
        if self.sem_emb_type == "complex_sem":
            query_ball = query_ball_point(self.radius, self.query_num, coordinates, coordinates, dis_mats)
            ball_features = index_points(x, query_ball)
            rel_pos = x[:, :, None] - ball_features
            combine = self.sem_trans(torch.cat([x[:, :, None].repeat(1, 1, rel_pos.shape[-2], 1), rel_pos], dim=-1))
            x = torch.cat([torch.mean(combine, dim=-2), torch.max(combine, dim=-2).values], dim=-1)
        return x


class Vector_Attention(nn.Module):
    def __init__(self, dim, pos_emb_type, size, quant_size):
        super(Vector_Attention, self).__init__()
        # feature transformation for inputs to query, key, and value
        self.qkv = nn.Linear(dim, 3 * dim)
        # mapping function to relation between query and key
        self.gamma = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        # position embedding type
        self.size = size
        self.quant_size = quant_size
        self.pos_emb_type = pos_emb_type
        if self.pos_emb_type == "complex_pos":
            quant_pillar_len_width, quant_pillar_height = int((2 * size + 1e-4) // quant_size), int((2 * 2 + 1e-4) // quant_size)
            self.table_x = nn.Parameter(torch.zeros(3, quant_pillar_len_width, dim))  # 0, 1, 2 for q, k, v's x off-set respectively
            self.table_y = nn.Parameter(torch.zeros(3, quant_pillar_len_width, dim))
            self.table_z = nn.Parameter(torch.zeros(3, quant_pillar_height, dim))
            nn.init.trunc_normal_(self.table_x, 0.02), nn.init.trunc_normal_(self.table_y, 0.02), nn.init.trunc_normal_(self.table_z, 0.02)
        elif self.pos_emb_type == "simple_pos":
            self.pos_enc = nn.Sequential(
                nn.Linear(3, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
            )
        self.down = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True)
        )

    def vec_att(self, feat, distance):
        N, dim = feat.shape
        qkv = self.qkv(feat).reshape(N, 3, dim).permute(1, 0, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        relation = self.gamma(q[:, None] - k[None])
        if self.pos_emb_type == "complex_pos":
            idx = torch.cat([((distance[:, :, :-1] + self.size) // self.quant_size).clamp(max=self.size * 2.0 / self.quant_size - 1),
                             ((distance[:, :, -1] + 2.0) // self.quant_size).clamp(max=2.0 * 2 / self.quant_size - 1)[:, :, None]], dim=-1).int()
            enc_q = self.table_x[0][idx[:, :, 0]] + self.table_y[0][idx[:, :, 1]] + self.table_z[0][idx[:, :, 2]]
            enc_k = self.table_x[1][idx[:, :, 0]] + self.table_y[1][idx[:, :, 1]] + self.table_z[1][idx[:, :, 2]]
            enc_v = self.table_x[2][idx[:, :, 0]] + self.table_y[2][idx[:, :, 1]] + self.table_z[2][idx[:, :, 2]]
            pos_bias = q[:, None] * enc_q + k[None] * enc_k
            relation += pos_bias
            v = v[None] + enc_v
        elif self.pos_emb_type == "simple_pos":
            pos_bias = self.pos_enc(distance)
            relation += pos_bias
            v = v[None] + pos_bias
        weights = F.softmax(relation, dim=1)
        att = torch.sum(weights * v, dim=1)
        return att

    def forward(self, feat, distance):
        patch_att = self.vec_att(feat, distance)
        mean_rep = torch.mean(patch_att, dim=0)[None].repeat(patch_att.shape[0], 1)
        max_rep = torch.max(patch_att, dim=0).values[None].repeat(patch_att.shape[0], 1)
        x = torch.cat([patch_att, mean_rep, max_rep], dim=-1)
        return self.down(x)


class Pillar_Feature_Encoder(nn.Module):
    def __init__(self, layer_num, dim, pos_emb_type, size, quant_size, conv_dim):
        super(Pillar_Feature_Encoder, self).__init__()
        self.dim = dim
        self.size = size
        self.conv_dim = conv_dim
        self.table = {k: idx for k, idx in enumerate([(i, j) for i in range(int(2 / self.size)) for j in range(int(2 / self.size))])}
        self.pil_enc = nn.ModuleList([
            Vector_Attention(dim, pos_emb_type, size, quant_size) for _ in range(layer_num)
        ])
        self.trans = nn.Sequential(
            nn.Linear((layer_num + 1) * dim, (layer_num + 1) * dim),
            nn.LayerNorm((layer_num + 1) * dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, coordinates, groups, eff_groups):
        B, N, C = features.shape
        outputs = []
        for_retrieve = []
        for i in range(B):
            feature, coordinate, group, eff_group = features[i], coordinates[i], groups[i], eff_groups[i]
            output = torch.zeros([self.conv_dim, int(2 / self.size), int(2 / self.size)]).to(feature.device)
            for_ret = torch.zeros([feature.shape[0], self.conv_dim]).to(feature.device)
            for j in eff_group:
                patch = group[j]
                feat = feature[patch]
                x = feat
                distance = coordinate[patch][:, None] - coordinate[patch][None]
                for module in self.pil_enc:
                    feat = module(feat, distance)
                    x = torch.cat([x, feat], dim=-1)
                x = self.trans(x)
                pil_fea = torch.cat([torch.mean(x, dim=0), torch.max(x, dim=0).values], dim=-1)
                output[:, self.table[j][1], self.table[j][0]] = pil_fea
                for_ret[patch] = pil_fea
            outputs.append(output)
            for_retrieve.append(for_ret)
        return torch.stack(outputs, dim=0), torch.stack(for_retrieve, dim=0)


class Residual_Depth_Wise(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Residual_Depth_Wise, self).__init__()
        self.point_wise = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, groups=1)
        self.bn_point = nn.BatchNorm2d(out_channel)
        self.relu_point = nn.ReLU()
        self.depth_wise = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channel)
        self.short_cut = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channel)
        self.bn_depth = nn.BatchNorm2d(out_channel)
        self.relu_depth = nn.ReLU(inplace=True)

    def forward(self, features):
        x = self.relu_point(self.bn_point(self.point_wise(features)))
        x = self.relu_depth(self.bn_depth(self.depth_wise(x) + self.short_cut(features)))
        return x


class Convolution_Aggregator(nn.Module):
    def __init__(self, settings):
        super(Convolution_Aggregator, self).__init__()
        self.seq = nn.Sequential(*[
            Residual_Depth_Wise(setting["in_channel"], setting["out_channel"], setting["kernel_size"], setting["stride"], setting["padding"]) for setting in settings
        ])
        self.point_wise = nn.Conv2d(in_channels=settings[-1]["out_channel"], out_channels=settings[-1]["out_channel"], kernel_size=1, stride=1, padding=0, groups=1)
        self.bn = nn.BatchNorm2d(settings[-1]["out_channel"])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        return self.relu(self.bn(self.point_wise(self.seq(features))))


class Information_Retrieve(nn.Module):
    def __init__(self, info_ret_dim):
        super(Information_Retrieve, self).__init__()
        self.trans = nn.Sequential(
            nn.Linear(info_ret_dim, info_ret_dim),
            nn.LayerNorm(info_ret_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, sem_embed_bat, for_retrieve_bat, conv_aggr_bat, classes=None):
        B, N, _ = sem_embed_bat.shape
        if classes is not None:
            conv_aggr_bat = torch.cat([conv_aggr_bat.reshape(B, -1), classes], dim=-1)
        clo_fea_rep = conv_aggr_bat.reshape(B, 1, -1).repeat(1, N, 1)
        return self.trans(torch.cat([sem_embed_bat, for_retrieve_bat, clo_fea_rep], dim=-1))


class Rec_Head(nn.Module):
    def __init__(self, num_category, rec_head_dim, drop_rate):
        super(Rec_Head, self).__init__()
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(rec_head_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_category)
        )

    def forward(self, features):
        return self.head(torch.cat([torch.mean(features, dim=1), torch.max(features, dim=1).values], dim=-1))


class Par_Head(nn.Module):
    def __init__(self, num_part, par_head_dim, drop_rate):
        super(Par_Head, self).__init__()
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(par_head_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_part)
        )

    def forward(self, features):
        return self.head(features)


class Sem_Head(nn.Module):
    def __init__(self, num_category, sem_head_dim, drop_rate):
        super(Sem_Head, self).__init__()
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(sem_head_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_category)
        )

    def forward(self, features):
        return self.head(features)


class Unifier_Cls(nn.Module):
    def __init__(self, sem_emb_type, init_dim, dim, radius, query_num, layer_num, pos_emb_type, size, quant_size, conv_dim, settings, info_ret_dim, num_category, rec_head_dim, drop_rate):
        super(Unifier_Cls, self).__init__()
        self.sem_embed = Semantic_Embedding(sem_emb_type=sem_emb_type, init_dim=init_dim, dim=dim, radius=radius, query_num=query_num)
        self.pil_fea_enc = Pillar_Feature_Encoder(layer_num=layer_num, dim=dim, pos_emb_type=pos_emb_type, size=size, quant_size=quant_size, conv_dim=conv_dim)
        self.conv_aggr = Convolution_Aggregator(settings=settings)
        self.info_ret = Information_Retrieve(info_ret_dim=info_ret_dim)
        self.rec_head = Rec_Head(num_category=num_category, rec_head_dim=rec_head_dim, drop_rate=drop_rate)

        self.apply(_init_vit_weights)

    def forward(self, features, dis_mats, coordinates, groups, eff_groups):
        sem_embed_bat = self.sem_embed(features, coordinates, dis_mats)
        pil_fea_enc_bat, for_retrieve_bat = self.pil_fea_enc(sem_embed_bat, coordinates, groups, eff_groups)
        conv_aggr_bat = self.conv_aggr(pil_fea_enc_bat)
        inf_ret = self.info_ret(sem_embed_bat, for_retrieve_bat, conv_aggr_bat)
        x = self.rec_head(inf_ret)
        return x


class Unifier_Par(nn.Module):
    def __init__(self, sem_emb_type, init_dim, dim, radius, query_num, layer_num, pos_emb_type, size, quant_size, conv_dim, settings, info_ret_dim, num_part, par_head_dim, drop_rate):
        super(Unifier_Par, self).__init__()
        self.sem_embed = Semantic_Embedding(sem_emb_type=sem_emb_type, init_dim=init_dim, dim=dim, radius=radius, query_num=query_num)
        self.pil_fea_enc = Pillar_Feature_Encoder(layer_num=layer_num, dim=dim, pos_emb_type=pos_emb_type, size=size, quant_size=quant_size, conv_dim=conv_dim)
        self.conv_aggr = Convolution_Aggregator(settings=settings)
        self.info_ret = Information_Retrieve(info_ret_dim=info_ret_dim)
        self.par_head = Par_Head(num_part=num_part, par_head_dim=par_head_dim, drop_rate=drop_rate)

        self.apply(_init_vit_weights)

    def forward(self, features, dis_mats, coordinates, groups, eff_groups, classes):
        sem_embed_bat = self.sem_embed(features, coordinates, dis_mats)
        pil_fea_enc_bat, for_retrieve_bat = self.pil_fea_enc(sem_embed_bat, coordinates, groups, eff_groups)
        conv_aggr_bat = self.conv_aggr(pil_fea_enc_bat)
        inf_ret = self.info_ret(sem_embed_bat, for_retrieve_bat, conv_aggr_bat, classes)
        res = self.par_head(inf_ret)
        return res


class Unifier_Sem(nn.Module):
    def __init__(self, sem_emb_type, init_dim, dim, radius, query_num, layer_num, pos_emb_type, size, quant_size, conv_dim, settings, info_ret_dim, num_category, sem_head_dim, drop_rate):
        super(Unifier_Sem, self).__init__()
        self.sem_embed = Semantic_Embedding(sem_emb_type=sem_emb_type, init_dim=init_dim, dim=dim, radius=radius, query_num=query_num)
        self.pil_fea_enc = Pillar_Feature_Encoder(layer_num=layer_num, dim=dim, pos_emb_type=pos_emb_type, size=size, quant_size=quant_size, conv_dim=conv_dim)
        self.conv_aggr = Convolution_Aggregator(settings=settings)
        self.info_ret = Information_Retrieve(info_ret_dim=info_ret_dim)
        self.sem_head = Sem_Head(num_category=num_category, sem_head_dim=sem_head_dim, drop_rate=drop_rate)

        self.apply(_init_vit_weights)

    def forward(self, features, dis_mats, coordinates, groups, eff_groups):
        sem_embed_bat = self.sem_embed(features, coordinates, dis_mats)
        pil_fea_enc_bat, for_retrieve_bat = self.pil_fea_enc(sem_embed_bat, coordinates, groups, eff_groups)
        conv_aggr_bat = self.conv_aggr(pil_fea_enc_bat)
        inf_ret = self.info_ret(sem_embed_bat, for_retrieve_bat, conv_aggr_bat)
        res = self.sem_head(inf_ret)
        return res


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def get_unifier_cls(num_category, size, init_dim, sem_emb_type="complex_sem", dim=32, layer_num=2, query_num=30, pos_emb_type="complex_pos", quant_size=0.02, drop_rate=0.5):
    radius = size
    conv_dim = (layer_num + 1) * dim * 2
    info_ret_dim = conv_dim * 2 + dim
    rec_head_dim = info_ret_dim * 2

    # (in_channel - kernel_size + 2padding) / stride + 1

    # size = 0.1
    # settings = [
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 10, "stride": 2, "padding": 1},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    # ]

    size = 0.2
    settings = [
        {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 6, "stride": 1, "padding": 0},
        {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
        {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    ]

    # size = 0.25
    # settings = [
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 4, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    # ]

    # size = 0.4
    # settings = [
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    # ]

    return Unifier_Cls(sem_emb_type=sem_emb_type, init_dim=init_dim, dim=dim, radius=radius, query_num=query_num, layer_num=layer_num, pos_emb_type=pos_emb_type,
                       size=size, quant_size=quant_size, conv_dim=conv_dim, settings=settings, info_ret_dim=info_ret_dim, num_category=num_category,
                       rec_head_dim=rec_head_dim, drop_rate=drop_rate)


def get_unifier_par(num_part, size, init_dim, num_classes, sem_emb_type="complex_sem", dim=32, layer_num=3, query_num=50, pos_emb_type="complex_pos", quant_size=0.02, drop_rate=0.5):
    radius = size
    conv_dim = (layer_num + 1) * dim * 2
    info_ret_dim = conv_dim * 2 + dim + num_classes
    par_head_dim = info_ret_dim

    # size = 0.1
    # settings = [
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 10, "stride": 2, "padding": 1},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    # ]

    # size = 0.2
    # settings = [
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 6, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    # ]

    size = 0.25
    settings = [
        {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 4, "stride": 1, "padding": 0},
        {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
        {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    ]

    # size = 0.4
    # settings = [
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    # ]

    return Unifier_Par(sem_emb_type=sem_emb_type, init_dim=init_dim, dim=dim, radius=radius, query_num=query_num, layer_num=layer_num, pos_emb_type=pos_emb_type,
                       size=size, quant_size=quant_size, conv_dim=conv_dim, settings=settings, info_ret_dim=info_ret_dim, num_part=num_part,
                       par_head_dim=par_head_dim, drop_rate=drop_rate)


def get_unifier_sem(num_category, size, init_dim, sem_emb_type="complex_sem", dim=32, layer_num=4, query_num=50, pos_emb_type="complex_pos", quant_size=0.02, drop_rate=0.5):
    radius = size
    conv_dim = (layer_num + 1) * dim * 2
    info_ret_dim = conv_dim * 2 + dim
    sem_head_dim = info_ret_dim

    size = 0.1
    settings = [
        {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 10, "stride": 2, "padding": 1},
        {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
        {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
        {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    ]

    # size = 0.2
    # settings = [
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 6, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    # ]

    # size = 0.25
    # settings = [
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 4, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    # ]

    # size = 0.4
    # settings = [
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    #     {"in_channel": conv_dim, "out_channel": conv_dim, "kernel_size": 3, "stride": 1, "padding": 0},
    # ]

    return Unifier_Sem(sem_emb_type=sem_emb_type, init_dim=init_dim, dim=dim, radius=radius, query_num=query_num, layer_num=layer_num, pos_emb_type=pos_emb_type,
                       size=size, quant_size=quant_size, conv_dim=conv_dim, settings=settings, info_ret_dim=info_ret_dim, num_category=num_category,
                       sem_head_dim=sem_head_dim, drop_rate=drop_rate)
