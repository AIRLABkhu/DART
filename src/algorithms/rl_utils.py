import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from captum.attr import GuidedBackprop, GuidedGradCam


class HookFeatures:
    def __init__(self, module):
        self.feature_hook = module.register_forward_hook(self.feature_hook_fn)

    def feature_hook_fn(self, module, input, output):
        self.features = output.clone().detach()
        self.gradient_hook = output.register_hook(self.gradient_hook_fn)

    def gradient_hook_fn(self, grad):
        self.gradients = grad

    def close(self):
        self.feature_hook.remove()
        self.gradient_hook.remove()


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, action=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.action = action

    def forward(self, obs):
        if self.action is None:
            return self.model(obs)[0]
        return self.model(obs, self.action)[0]


def compute_guided_backprop(obs, action, model):
    model = ModelWrapper(model, action=action)
    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(obs)
    return attribution

def compute_guided_gradcam(obs, action, model):
    obs.requires_grad_()
    obs.retain_grad()
    model = ModelWrapper(model, action=action)
    gbp = GuidedGradCam(model,layer=model.model.encoder.head_cnn.layers)
    attribution = gbp.attribute(obs,attribute_to_layer_input=True)
    return attribution

def compute_vanilla_grad(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    q, q2 = critic_target(obs, action.detach())
    q.sum().backward()
    return obs.grad



def compute_attribution(model, obs, action=None,method="guided_backprop"):
    if method == "guided_backprop":
        return compute_guided_backprop(obs, action, model)
    if method == 'guided_gradcam':
        return compute_guided_gradcam(obs,action,model)
    return compute_vanilla_grad(model, obs, action)


def compute_features_attribution(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    hook = HookFeatures(critic_target.encoder)
    q, _ = critic_target(obs, action.detach())
    q.sum().backward()
    features_gardients = hook.gradients
    hook.close()
    return obs.grad, features_gardients

def compute_attribution2(model, obs, action=None):
    # Create tensors for t-2, t-1, t
    o1 = torch.cat((obs[:, 0:3, :, :], obs[:, 0:3, :, :], obs[:, 0:3, :, :]), dim=1)
    o2 = torch.cat((obs[:, 3:6, :, :], obs[:, 3:6, :, :], obs[:, 3:6, :, :]), dim=1)
    o3 = torch.cat((obs[:, 6:9, :, :], obs[:, 6:9, :, :], obs[:, 6:9, :, :]), dim=1)
    
    combined_obs = torch.cat((o1, o2, o3), dim=0)
    combined_obs.requires_grad_(True)
    combined_action=torch.cat((action,action,action),dim=0)

    #print(combined_obs.shape,combined_action.shape)
    combined_grad = compute_guided_backprop(combined_obs, combined_action,model)
    o1_grad, o2_grad, o3_grad = torch.split(combined_grad, obs.size(0), dim=0)
    
    o_grad = torch.zeros_like(o1_grad)
    
    o_grad[:, 0:3, :, :] = o1_grad[:, 6:9, :, :]
    o_grad[:, 3:6, :, :] = o2_grad[:, 6:9, :, :]
    o_grad[:, 6:9, :, :] = o3_grad[:, 6:9, :, :]

    return o_grad


def compute_attribution_mask(obs_grad, quantile=0.95):
    mask = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        q = torch.quantile(attributions.flatten(1), quantile, 1)
        mask.append((attributions >= q[:, None, None]).unsqueeze(1).repeat(1, 3, 1, 1))
    return torch.cat(mask, dim=1).float()

def compute_attribution_mask2(obs_grad):
    mask1 = []
    mask2=[]
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        #attributions = obs_grad[:, i : i + 3].abs().mean(dim=1)
        # Flatten the attributions and apply Min-Max normalization
        flat_attributions = attributions.flatten(1)
        min_vals = flat_attributions.min(dim=1, keepdim=True)[0]
        max_vals = flat_attributions.max(dim=1, keepdim=True)[0]
        normalized_attributions=flat_attributions
        normalized_attributions = (flat_attributions - min_vals) / (max_vals - min_vals)

        mean = torch.mean(normalized_attributions, dim=1, keepdim=True)
        median=torch.median(normalized_attributions, dim=1, keepdim=True).values

        normalized_attributions1 = (normalized_attributions>mean).float().view(attributions.shape)
        normalized_attributions2 = (normalized_attributions>median).float().view(attributions.shape)


        mask1.append((normalized_attributions1).unsqueeze(1).repeat(1, 3, 1, 1))
        mask2.append((normalized_attributions2).unsqueeze(1).repeat(1, 3, 1, 1))

    return torch.cat(mask1, dim=1),torch.cat(mask2, dim=1)


def compute_attribution_mask3(obs_grad):
    mask1 = []
    mask2 = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        # Flatten the attributions and apply Min-Max normalization
        flat_attributions = attributions.flatten(1)
        min_vals = flat_attributions.min(dim=1, keepdim=True)[0]
        max_vals = flat_attributions.max(dim=1, keepdim=True)[0]
        normalized_attributions = flat_attributions
        normalized_attributions = (flat_attributions - min_vals) / (max_vals - min_vals)

        mean = torch.mean(normalized_attributions, dim=1, keepdim=True)
        median = torch.median(normalized_attributions, dim=1, keepdim=True).values

        # Create masks where the normalized attribution is greater than the mean or median
        # Set to 1 if condition is met, otherwise retain the original pixel value
        mask1_attributions = torch.where(normalized_attributions > mean, torch.ones_like(normalized_attributions), torch.log(1+normalized_attributions)).view(attributions.shape)
        mask2_attributions = torch.where(normalized_attributions > median, torch.ones_like(normalized_attributions), torch.log(1+normalized_attributions)).view(attributions.shape)
        mask1.append((mask1_attributions).unsqueeze(1).repeat(1, 3, 1, 1))
        mask2.append((mask2_attributions).unsqueeze(1).repeat(1, 3, 1, 1))

    return torch.cat(mask1, dim=1).float(), torch.cat(mask2, dim=1).float()



def long_tailed_mapping_linear(values):

    # Compute the mean of the input values
    mean = torch.mean(values)
    values=torch.clamp(values,min=1e-128)
    min_val = values.min()
    max_val = values.max()  # Use mean as the upper bound for the mapping
    mapped_values = (values - min_val) / (max_val - min_val)
    
    return mapped_values

def long_tailed_mapping_log(values):

    # Compute the mean of the input values
    mean = torch.mean(values)
    
    values=torch.clamp(values,min=1e-6)
    values=torch.log(values)
    min_val = values.min()
    max_val = values.max()  # Use mean as the upper bound for the mapping
    mapped_values = (values - min_val) / (max_val - min_val)
    
    return mapped_values

def long_tailed_mapping_exp(values,v):

    # Compute the mean of the input values
    #mean = torch.mean(values)
    
    values=values**v
    min_val = values.min()
    max_val = values.max()  # Use mean as the upper bound for the mapping
    mapped_values = (values - min_val) / (max_val - min_val)
    
    return mapped_values

def compute_attribution_mask_exp(obs_grad,v):
    mask1 = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        # Flatten the attributions and apply Min-Max normalization
        flat_attributions = attributions.flatten(1)
        mask1_attributions = flat_attributions

        min_vals = mask1_attributions.min(dim=1, keepdim=True)[0]
        max_vals = mask1_attributions.max(dim=1, keepdim=True)[0]
        normalized_attributions = mask1_attributions
        normalized_attributions = (mask1_attributions - min_vals) / (max_vals - min_vals)
        normalized_attributions=long_tailed_mapping_exp(normalized_attributions,v)
        normalized_attributions=normalized_attributions.view(attributions.shape)


        mask1.append((normalized_attributions).unsqueeze(1).repeat(1, 3, 1, 1))

    return torch.cat(mask1, dim=1).float()


def compute_attribution_mask_log(obs_grad):
    mask1 = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        # Flatten the attributions and apply Min-Max normalization
        flat_attributions = attributions.flatten(1)
        mask1_attributions = flat_attributions

        min_vals = mask1_attributions.min(dim=1, keepdim=True)[0]
        max_vals = mask1_attributions.max(dim=1, keepdim=True)[0]
        normalized_attributions = mask1_attributions
        normalized_attributions = (mask1_attributions - min_vals) / (max_vals - min_vals)
        normalized_attributions=long_tailed_mapping_log(normalized_attributions)
        normalized_attributions=normalized_attributions.view(attributions.shape)


        mask1.append((normalized_attributions).unsqueeze(1).repeat(1, 3, 1, 1))

    return torch.cat(mask1, dim=1).float()

def compute_attribution_mask_linear(obs_grad):
    mask1 = []
    mask2 = []
    m=torch.nn.Softmax()
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        # Flatten the attributions and apply Min-Max normalization
        flat_attributions = attributions.flatten(1)
        mask1_attributions = m(flat_attributions)

        min_vals = mask1_attributions.min(dim=1, keepdim=True)[0]
        max_vals = mask1_attributions.max(dim=1, keepdim=True)[0]
        normalized_attributions = mask1_attributions
        normalized_attributions = (mask1_attributions - min_vals) / (max_vals - min_vals)
        normalized_attributions=long_tailed_mapping_linear(normalized_attributions)
        normalized_attributions=normalized_attributions.view(attributions.shape)


        mask1.append((normalized_attributions).unsqueeze(1).repeat(1, 3, 1, 1))

    return torch.cat(mask1, dim=1).float()

def make_obs_grid(obs, n=4):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            sample.append(obs[i, j : j + 3].unsqueeze(0))
    sample = torch.cat(sample, 0)
    return make_grid(sample, nrow=3) / 255.0


def make_attribution_pred_grid(attribution_pred, n=4):
    return make_grid(attribution_pred[:n], nrow=1)


def make_obs_grad_grid(obs_grad, n=4):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            channel_attribution, _ = torch.max(obs_grad[i, j : j + 3], dim=0)
            sample.append(channel_attribution[(None,) * 2] / channel_attribution.max())
    sample = torch.cat(sample, 0)
    q = torch.quantile(sample.flatten(1), 0.97, 1)
    sample[sample <= q[:, None, None, None]] = 0
    return make_grid(sample, nrow=3)



def compute_guided_backprop_viz(model, obs, action):
    model = ModelWrapper(model, action=action)
    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(obs)
    obs_grad=attribution
    
    mask=[]
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        # attributions = obs_grad[:, i : i + 3].abs().min(dim=1)[0]
        flat_attributions = attributions.flatten(1)
        min_vals = flat_attributions.min(dim=1, keepdim=True)[0]
        max_vals = flat_attributions.max(dim=1, keepdim=True)[0]
        normalized_attributions=flat_attributions
        normalized_attributions = (flat_attributions - min_vals) / (max_vals - min_vals)
        normalized_attributions=normalized_attributions.view(attributions.shape)
        mask.append((normalized_attributions).unsqueeze(1).repeat(1, 1, 1, 1))


    return torch.cat(mask,dim=1)
def compute_attribution2_viz(model, obs, action=None):
    # Create tensors for t-2, t-1, t
    o_grad=compute_attribution2(model,obs,action)

    mask=[]
    for i in [0, 3, 6]:
        attributions = o_grad[:, i : i + 3].abs().max(dim=1)[0]
        # attributions = o_grad[:, i : i + 3].abs().min(dim=1)[0]
        flat_attributions = attributions.flatten(1)
        min_vals = flat_attributions.min(dim=1, keepdim=True)[0]
        max_vals = flat_attributions.max(dim=1, keepdim=True)[0]
        normalized_attributions=flat_attributions
        normalized_attributions = (flat_attributions - min_vals) / (max_vals - min_vals)
        normalized_attributions=normalized_attributions.view(attributions.shape)
        mask.append((normalized_attributions).unsqueeze(1).repeat(1, 1, 1, 1))


    return torch.cat(mask,dim=1)



def gradient_distribution_viz(model, obs, action):
    model = ModelWrapper(model, action=action)
    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(obs)
    obs_grad=attribution
    
    # Lists to hold flattened attributions for each channel
    flat_attributions_list = []
    labels = []

    for i, channel in enumerate([0, 3, 6]):
        attributions = obs_grad[:, channel:channel + 3].abs().max(dim=1)[0]
        flat_attributions = attributions.flatten(1)
        flat_attributions_list.append(flat_attributions)
        labels.append(f't {i -2}')

    # Concatenate all attributions
    concatenated_attributions = torch.cat(flat_attributions_list, dim=1)

    # Normalize all concatenated attributions
    min_vals = concatenated_attributions.min()
    max_vals = concatenated_attributions.max()

    # Plot histograms
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red']

    for i, flat_attributions in enumerate(flat_attributions_list):
        # Calculate normalized data based on the overall normalization
        data=flat_attributions
        data = (flat_attributions - min_vals) / (max_vals - min_vals)
        
        data=data[data>0.2]
        plt.hist(data.flatten().cpu().numpy(), bins=50, color=colors[i], alpha=0.7, label=labels[i])
        plt.xlim(0.2,1.0)

    plt.title('Distribution of Normalized Attributions for each time')
    plt.xlabel('Normalized Attribution')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def gradient_distribution_viz2(model, obs, action):
    obs_grad=compute_attribution2(model,obs,action)
    
    # Lists to hold flattened attributions for each channel
    flat_attributions_list = []
    labels = []

    for i, channel in enumerate([0, 3, 6]):
        attributions = obs_grad[:, channel:channel + 3].abs().max(dim=1)[0]
        flat_attributions = attributions.flatten(1)
        flat_attributions_list.append(flat_attributions)
        labels.append(f't {i -2}')

    # Concatenate all attributions
    concatenated_attributions = torch.cat(flat_attributions_list, dim=1)

    # Normalize all concatenated attributions
    min_vals = concatenated_attributions.min()
    max_vals = concatenated_attributions.max()

    # Plot histograms
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red']

    for i, flat_attributions in enumerate(flat_attributions_list):
        # Calculate normalized data based on the overall normalization
        data=flat_attributions
        data = (flat_attributions - min_vals) / (max_vals - min_vals)
        
        data=data[data>0.2]
        plt.xlim(0.2,1.0)
        plt.hist(data.flatten().cpu().numpy(), bins=50, color=colors[i], alpha=0.7, label=labels[i])
    
    plt.title('Distribution of Normalized Attributions for each time')
    plt.xlabel('Normalized Attribution')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def log_weight_distribution(model, obs, action):
    model = ModelWrapper(model, action=action)
    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(obs)
    obs_grad=attribution
    
    # Lists to hold flattened attributions for each channel
    flat_attributions_list = []
    labels = []

    for i, channel in enumerate([0, 3, 6]):
        attributions = obs_grad[:, channel:channel + 3].abs().max(dim=1)[0]
        flat_attributions = attributions.flatten(1)
        flat_attributions_list.append(flat_attributions)
        labels.append(f't {i -2}')

    # Concatenate all attributions
    concatenated_attributions = torch.cat(flat_attributions_list, dim=1)

    # Normalize all concatenated attributions
    min_vals = concatenated_attributions.min()
    max_vals = concatenated_attributions.max()

    # Plot histograms
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red']
    
    for i, flat_attributions in enumerate(flat_attributions_list):
        # Calculate normalized data based on the overall normalization
        data=flat_attributions
        data = (flat_attributions - min_vals) / (max_vals - min_vals)
        data=long_tailed_mapping_log(data)
        
        #data=data[data>0.01]
        plt.hist(data.flatten().cpu().numpy(), bins=50, color=colors[i], alpha=0.7, label=labels[i])
    
    plt.title('Distribution of log weight')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def log_weight_distribution2(model, obs, action):
    obs_grad=compute_attribution2(model,obs,action)
    
    # Lists to hold flattened attributions for each channel
    flat_attributions_list = []
    labels = []

    for i, channel in enumerate([0, 3, 6]):
        attributions = obs_grad[:, channel:channel + 3].abs().max(dim=1)[0]
        flat_attributions = attributions.flatten(1)
        flat_attributions_list.append(flat_attributions)
        labels.append(f't {i -2}')

    # Concatenate all attributions
    concatenated_attributions = torch.cat(flat_attributions_list, dim=1)

    # Normalize all concatenated attributions
    min_vals = concatenated_attributions.min()
    max_vals = concatenated_attributions.max()

    # Plot histograms
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red']
    
    for i, flat_attributions in enumerate(flat_attributions_list):
        # Calculate normalized data based on the overall normalization
        data=flat_attributions
        data = (flat_attributions - min_vals) / (max_vals - min_vals)
        data=long_tailed_mapping_log(data)
        
        #data=data[data>0.01]
        plt.hist(data.flatten().cpu().numpy(), bins=50, color=colors[i], alpha=0.7, label=labels[i])
    
    plt.title('Distribution of log weight')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def linear_weight_distribution(model, obs, action):
    model = ModelWrapper(model, action=action)
    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(obs)
    obs_grad=attribution
    
    # Lists to hold flattened attributions for each channel
    flat_attributions_list = []
    labels = []

    for i, channel in enumerate([0, 3, 6]):
        attributions = obs_grad[:, channel:channel + 3].abs().max(dim=1)[0]
        flat_attributions = attributions.flatten(1)
        flat_attributions_list.append(flat_attributions)
        labels.append(f't {i -2}')

    # Concatenate all attributions
    concatenated_attributions = torch.cat(flat_attributions_list, dim=1)

    # Normalize all concatenated attributions
    min_vals = concatenated_attributions.min()
    max_vals = concatenated_attributions.max()

    # Plot histograms
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red']
    
    for i, flat_attributions in enumerate(flat_attributions_list):
        # Calculate normalized data based on the overall normalization
        data=flat_attributions
        data = (flat_attributions - min_vals) / (max_vals - min_vals)
        data=long_tailed_mapping_linear(data)
        
        #data=data[data>0.01]
        plt.hist(data.flatten().cpu().numpy(), bins=50, color=colors[i], alpha=0.7, label=labels[i])
    
    plt.title('Distribution of linear weight')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def linear_weight_distribution2(model, obs, action):
    obs_grad=compute_attribution2(model,obs,action)
    
    # Lists to hold flattened attributions for each channel
    flat_attributions_list = []
    labels = []

    for i, channel in enumerate([0, 3, 6]):
        attributions = obs_grad[:, channel:channel + 3].abs().max(dim=1)[0]
        flat_attributions = attributions.flatten(1)
        flat_attributions_list.append(flat_attributions)
        labels.append(f't {i -2}')

    # Concatenate all attributions
    concatenated_attributions = torch.cat(flat_attributions_list, dim=1)

    # Normalize all concatenated attributions
    min_vals = concatenated_attributions.min()
    max_vals = concatenated_attributions.max()

    # Plot histograms
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red']
    
    for i, flat_attributions in enumerate(flat_attributions_list):
        # Calculate normalized data based on the overall normalization
        data=flat_attributions
        data = (flat_attributions - min_vals) / (max_vals - min_vals)
        data=long_tailed_mapping_linear(data)
        
        #data=data[data>0.01]
        plt.hist(data.flatten().cpu().numpy(), bins=50, color=colors[i], alpha=0.7, label=labels[i])
    
    plt.title('Distribution of linear weight')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()