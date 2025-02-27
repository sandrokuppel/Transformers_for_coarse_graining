import torch
import torch.nn.functional as F
from ase import Atoms
from ase.io import read
from dscribe.descriptors import SOAP
from mace.calculators import mace_mp
from matplotlib import pyplot as plt
from rdkit2ase import smiles2atoms
from torch import nn


def calc_center_of_mass(pos, DFT_order = False, batched = True):
    if batched:
        if DFT_order:
            m_center = pos[:,[0, 3,4],:]*torch.tensor([12,1,1])[None,:,None]
            m_side1 = pos[:,[1, 5,7,8],:]*torch.tensor([12,1,1,1])[None,:,None]
            m_side2 = pos[:,[2, 6,9,10],:]*torch.tensor([12,1,1,1])[None,:,None]
        else:
            m_center = pos[:,[1, 6,7],:]*torch.tensor([12,1,1])[None,:,None]
            m_side1 = pos[:,[0, 3,4,5],:]*torch.tensor([12,1,1,1])[None,:,None]
            m_side2 = pos[:,[2, 8,9,10],:]*torch.tensor([12,1,1,1])[None,:,None]
        cm_c = torch.sum(m_center, dim=1)/14
        cm_s1 = torch.sum(m_side1, dim=1)/15
        cm_s2 = torch.sum(m_side2, dim=1)/15
    else:
        if DFT_order:
            m_center = pos[[0, 3,4],:]*torch.tensor([12,1,1])[:,None]
            m_side1 = pos[[1, 5,7,8],:]*torch.tensor([12,1,1,1])[:,None]
            m_side2 = pos[[2, 6,9,10],:]*torch.tensor([12,1,1,1])[:,None]
        else:
            m_center = pos[[1, 6,7],:]*torch.tensor([12,1,1])[:,None]
            m_side1 = pos[[0, 3,4,5],:]*torch.tensor([12,1,1,1])[:,None]
            m_side2 = pos[[2, 8,9,10],:]*torch.tensor([12,1,1,1])[:,None]
        cm_c = torch.sum(m_center, dim=0)/14
        cm_s1 = torch.sum(m_side1, dim=0)/15
        cm_s2 = torch.sum(m_side2, dim=0)/15
    return cm_c, cm_s1, cm_s2

def distillation(student_scores, y, teacher_scores, labels, T, alpha=1):
    p = F.log_softmax(student_scores/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_mse = F.mse_loss(y, labels)
    return l_kl * alpha + l_mse * (1. - alpha)


class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_lr, epochs, steps_per_epoch, pct_start, div_factor, final_div_factor, last_epoch=-1):
        self.total_steps = int(epochs*steps_per_epoch)
        self.warmup_steps = int(self.total_steps*pct_start)
        self.init_lr = max_lr/div_factor
        self.final_lr = max_lr/final_div_factor
        self.max_lr = max_lr
        self.lr_gab_warmup = max_lr-self.init_lr
        self.lr_gab_decay = max_lr-self.final_lr
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [self.init_lr + self.lr_gab_warmup * step / self.warmup_steps]
            return [max_lr/base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [self.max_lr - self.lr_gab_decay * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)]
            return [base_lr * (1 - (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)) for base_lr in self.base_lrs]

def rename_keys(state_dict, rename_func, prefix):
    return {rename_func(prefix, key): value for key, value in state_dict.items()}
def prepend_prefix(prefix,key):
    return prefix + key
def remove_prefix(prefix,key):
    return key.replace(prefix, '', 1)
def remove_keys(state_dict, keys_to_remove):
    return {key: value for key, value in state_dict.items() if key not in keys_to_remove}


# creates positional encodings for a sequence of length max_sequence_length and dimension d_model
# positional encoding like in paper "Attention is all you need"
class Positionalencoding(nn.Module):
    """
    Positional Encoding Class
    
    Created position encoding like propoesed in the paper "Attention is all you need"
    """
    def __init__(self, max_sequence_length, d_model):
        """
        Initializes the Positional Encoding class
        
        Parameters:
        ------------
        d_model : int
            model dimension
        max_sequence_length : int
            maximum sequence length
        """
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model 
        
    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

def create_test_array_position_CG(dtype=torch.float32, DFT_order = False, path = None):
    macemp = mace_mp() 
    atoms = read(path, index=0)
    atoms.calc = macemp
    positions = atoms.get_positions()
    step = 0.005
    y_range = 0.7
    if DFT_order:
        positions[0,1] = positions[0,1] - y_range/2
        positions[3,1] = positions[3,1] - y_range/2
        positions[4,1] = positions[4,1] - y_range/2
    else:
        positions[1,1] = positions[1,1] - y_range/2
        positions[6,1] = positions[6,1] - y_range/2
        positions[7,1] = positions[7,1] - y_range/2
    y_pos = []
    n = int(y_range/step)
    energies_mace = []
    force_mace = []
    pos = []
    for i in range(n):
        if DFT_order:
            positions[0,1] = positions[0,1] + step
            positions[3,1] = positions[3,1] + step
            positions[4,1] = positions[4,1] + step
            y_pos.append(positions[0,1].item())
        else:
            positions[1,1] = positions[1,1] + step
            positions[6,1] = positions[6,1] + step
            positions[7,1] = positions[7,1] + step
            y_pos.append(positions[1,1].item())
        atoms.set_positions(positions)
        # get energy and force with mace
        energies_mace.append(atoms.get_potential_energy())
        force = atoms.get_forces()
        if DFT_order:
            force_mace.append(torch.sum(torch.tensor(force, dtype=dtype)[[0,3,4],1],dim=0))
        else:
            force_mace.append(force[1,1])       # force only the C atom that is moved
        pos.append(torch.tensor(positions, dtype=dtype)[None,...].clone())
    pos_tensor = torch.concat(pos, dim=0)
    return pos_tensor, y_pos, energies_mace, force_mace

def create_test_array_position(dtype=torch.float32, DFT_order = False, path = None):
    macemp = mace_mp() 
    atoms = read(path, index=0)
    atoms.calc = macemp
    positions = atoms.get_positions()
    step = 0.005
    y_range = 0.7
    if DFT_order:
        positions[0,1] = positions[0,1] - y_range/2
        positions[3,1] = positions[3,1] - y_range/2
        positions[4,1] = positions[4,1] - y_range/2
    else:
        positions[1,1] = positions[1,1] - y_range/2
        positions[6,1] = positions[6,1] - y_range/2
        positions[7,1] = positions[7,1] - y_range/2
    y_pos = []
    n = int(y_range/step)
    energies_mace = []
    force_mace = []
    pos = []
    for i in range(n):
        if DFT_order:
            positions[0,1] = positions[0,1] + step
            positions[3,1] = positions[3,1] + step
            positions[4,1] = positions[4,1] + step
            y_pos.append(positions[0,1].item())
        else:
            positions[1,1] = positions[1,1] + step
            positions[6,1] = positions[6,1] + step
            positions[7,1] = positions[7,1] + step
            y_pos.append(positions[1,1].item())
        atoms.set_positions(positions)
        # get energy and force with mace
        energies_mace.append(atoms.get_potential_energy())
        force = atoms.get_forces()
        if DFT_order:
            force_mace.append(force[0,1])
        else:
            force_mace.append(force[1,1])       # force only the C atom that is moved
        pos.append(torch.tensor(positions, dtype=dtype)[None,...].clone())
    pos_tensor = torch.concat(pos, dim=0)
    return pos_tensor, y_pos, energies_mace, force_mace

def create_test_array_CG_SOAP(dtype=torch.float32, DFT_order = False, rcut = 4.0, n_max = 8, l_max = 8, sigma1 = 0.125, sigma2 = 0.5):
    macemp = mace_mp() 
    atoms=smiles2atoms('CCC')
    atoms.calc = macemp
    if dtype == torch.float64:
        data_type = "float64"
    if dtype == torch.float32:
        data_type = "float32"
    soap = SOAP(
    species=["C", "H"],
    r_cut=rcut,
    n_max=n_max,
    l_max=l_max,
    sigma=sigma1,
    periodic=False,
    dtype = data_type
    )
    soap2 = SOAP(
    species=["Sn"],
    r_cut=rcut,
    n_max=n_max,
    l_max=l_max,
    sigma=sigma2,
    periodic=False,
    dtype=data_type,
    )
    if DFT_order:
        positions = torch.tensor([
            [0.0000, 0.5863, -0.0000],  # C1
            [-1.2681, -0.2626, 0.0000],  # C2
            [1.2681, -0.2626, -0.0000],  # C3
            [2.1576, 0.3743, -0.0000],  # H7
            [-1.3271, -0.9014, 0.8800],  # H8
            [0.0000, 1.2449, 0.8760],  # H4 
            [-1.3271, -0.9014, -0.8800],  # H9
            [-0.0003, 1.2453, -0.8758],  # H5
            [-2.1576, 0.3742, 0.0000],  # H6
            [1.3271, -0.9014, -0.8800],  # H10
            [1.3272, -0.9014, 0.8800]  # H11
        ], dtype=dtype)
    # https://cccbdb.nist.gov/exp2x.asp?casno=74986&charge=0
    else:
        positions = torch.tensor([
            [-1.2681, -0.2626, 0.0000],  # C2
            [0.0000, 0.5863, -0.0000],  # C1
            [1.2681, -0.2626, -0.0000],  # C3
            [0.0000, 1.2449, 0.8760],  # H4
            [-0.0003, 1.2453, -0.8758],  # H5
            [-2.1576, 0.3742, 0.0000],  # H6
            [2.1576, 0.3743, -0.0000],  # H7
            [-1.3271, -0.9014, 0.8800],  # H8
            [-1.3271, -0.9014, -0.8800],  # H9
            [1.3271, -0.9014, -0.8800],  # H10
            [1.3272, -0.9014, 0.8800]  # H11
        ], dtype=dtype)
    
    atoms.set_positions(positions)

    step = 0.005
    y_range = 0.5
    if DFT_order:
        positions[0,1] = positions[0,1] - y_range/2
        positions[3,1] = positions[3,1] - y_range/2
        positions[4,1] = positions[4,1] - y_range/2
    else:
        positions[1,1] = positions[1,1] - y_range/2
        positions[6,1] = positions[6,1] - y_range/2
        positions[7,1] = positions[7,1] - y_range/2
    y_pos = []
    n = int(y_range/step)
    energies_mace = []
    force_mace = []
    deriv_cm = []
    desc_cm = []
    for i in range(n):
        if DFT_order:
            positions[0,1] = positions[0,1] + step
            positions[3,1] = positions[3,1] + step
            positions[4,1] = positions[4,1] + step
            y_pos.append(positions[0,1].item())
        else:
            positions[1,1] = positions[1,1] + step
            positions[6,1] = positions[6,1] + step
            positions[7,1] = positions[7,1] + step
            y_pos.append(positions[1,1].item())
        atoms.set_positions(positions)
        # get energy and force with mace
        energies_mace.append(atoms.get_potential_energy())
        force = atoms.get_forces()
        if DFT_order:
            force_mace.append(force[0,1])
        else:
            force_mace.append(force[1,1])       # force only the C atom that is moved
        cm_c, cm_s1, cm_s2 = calc_center_of_mass(positions, DFT_order=DFT_order, batched=False)
        pos_data_cm = torch.cat([cm_c[None,:], cm_s1[None,:], cm_s2[None,:]], dim=0)
        CM_atoms = Atoms('Sn3', positions=pos_data_cm)
        derivatives, descriptors = soap2.derivatives(CM_atoms)
        deriv_cm.append(torch.tensor(derivatives, dtype=dtype)[None,...])
        desc_cm.append(torch.tensor(descriptors, dtype=dtype)[None,...])
    deriv_cm_tensor = torch.concat(deriv_cm, dim=0)
    desc_cm_tensor = torch.concat(desc_cm, dim=0)
    return desc_cm_tensor, deriv_cm_tensor, y_pos, energies_mace, force_mace

def create_test_array(dtype=torch.float32, DFT = False, SOAP_2_times = True, rcut = 6.0, n_max=8, l_max=8, sigma=0.125, sigma2=0.5):
    macemp = mace_mp() 
    atoms=smiles2atoms('CCC')
    atoms.calc = macemp
    if dtype == torch.float64:
        data_type = "float64"
    if dtype == torch.float32:
        data_type = "float32"
    soap = SOAP(
    species=["C", "H"],
    r_cut=rcut,
    n_max=n_max,
    l_max=l_max,
    sigma=sigma,
    periodic=False,
    dtype = data_type
    )
    soap2 = SOAP(
    species=["Sn"],
    r_cut=rcut,
    n_max=n_max,
    l_max=l_max,
    sigma=sigma2,
    periodic=False,
    dtype=data_type,
    )
    if DFT:
        positions = torch.tensor([
            [0.0000, 0.5863, -0.0000],  # C1
            [-1.2681, -0.2626, 0.0000],  # C2
            [1.2681, -0.2626, -0.0000],  # C3
            [2.1576, 0.3743, -0.0000],  # H7
            [-1.3271, -0.9014, 0.8800],  # H8
            [0.0000, 1.2449, 0.8760],  # H4 
            [-1.3271, -0.9014, -0.8800],  # H9
            [-0.0003, 1.2453, -0.8758],  # H5
            [-2.1576, 0.3742, 0.0000],  # H6
            [1.3271, -0.9014, -0.8800],  # H10
            [1.3272, -0.9014, 0.8800]  # H11
        ], dtype=dtype)
    # https://cccbdb.nist.gov/exp2x.asp?casno=74986&charge=0
    else:
        positions = torch.tensor([
            [-1.2681, -0.2626, 0.0000],  # C2
            [0.0000, 0.5863, -0.0000],  # C1
            [1.2681, -0.2626, -0.0000],  # C3
            [0.0000, 1.2449, 0.8760],  # H4
            [-0.0003, 1.2453, -0.8758],  # H5
            [-2.1576, 0.3742, 0.0000],  # H6
            [2.1576, 0.3743, -0.0000],  # H7
            [-1.3271, -0.9014, 0.8800],  # H8
            [-1.3271, -0.9014, -0.8800],  # H9
            [1.3271, -0.9014, -0.8800],  # H10
            [1.3272, -0.9014, 0.8800]  # H11
        ], dtype=dtype)
    
    atoms.set_positions(positions)

    step = 0.005
    y_range = 0.5
    if DFT:
        positions[0,1] = positions[0,1] - y_range/2
        positions[3,1] = positions[3,1] - y_range/2
        positions[4,1] = positions[4,1] - y_range/2
    else:
        positions[1,1] = positions[1,1] - y_range/2
        positions[6,1] = positions[6,1] - y_range/2
        positions[7,1] = positions[7,1] - y_range/2
    y_pos = []
    n = int(y_range/step)
    pos = []
    energies_mace = []
    force_mace = []
    deriv = []
    desc = []
    deriv_c = []
    desc_c = []
    for i in range(n):
        if DFT:
            positions[0,1] = positions[0,1] + step
            positions[3,1] = positions[3,1] + step
            positions[4,1] = positions[4,1] + step
            y_pos.append(positions[0,1].item())
        else:
            positions[1,1] = positions[1,1] + step
            positions[6,1] = positions[6,1] + step
            positions[7,1] = positions[7,1] + step
            y_pos.append(positions[1,1].item())
        pos.append(positions[None,...].clone())
        
        atoms.set_positions(positions)
        # get energy and force with mace
        energies_mace.append(atoms.get_potential_energy())
        force = atoms.get_forces()
        if DFT:
            force_mace.append(force[0,1])
        else:
            force_mace.append(force[1,1])       # force only the C atom that is moved
        propane = Atoms('C3H8', positions=positions)
        C_atoms = Atoms('Sn3', positions=positions[:3,:])
        derivatives, descriptors = soap.derivatives(propane)
        deriv.append(torch.tensor(derivatives, dtype=dtype)[None,...])
        desc.append(torch.tensor(descriptors, dtype=dtype)[None,...])
        derivatives, descriptors = soap2.derivatives(C_atoms)
        deriv_c.append(torch.tensor(derivatives, dtype=dtype)[None,...])
        desc_c.append(torch.tensor(descriptors, dtype=dtype)[None,...])
    pos_tensor = torch.concat(pos, dim=0)
    deriv_tensor = torch.concat(deriv, dim=0)
    desc_tensor = torch.concat(desc, dim=0)
    deriv_c_tensor = torch.concat(deriv_c, dim=0)
    desc_c_tensor = torch.concat(desc_c, dim=0)
    if SOAP_2_times:
        return desc_c_tensor, deriv_c_tensor, deriv_tensor, desc_tensor, y_pos, energies_mace, force_mace
    else:
        return pos_tensor, deriv_tensor, desc_tensor, y_pos, energies_mace, force_mace