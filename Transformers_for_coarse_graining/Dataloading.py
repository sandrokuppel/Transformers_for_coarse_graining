import numpy as np
import torch
from ase import Atoms
from dscribe.descriptors import SOAP
from torch.utils.data import Dataset


def load_xyz_energy_force(file_path, dtype=torch.float32):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        data = []
        data_forces = []
        i = 0
        while i < len(lines):
            config = []
            forces = []
            # The first line is the number of atoms
            num_atoms = int(lines[i].strip())
            i += 1
            
            # The second line is a comment line
            comment = lines[i].strip()
            i += 1
            
            # The next num_atoms lines are the atomic data
            for _ in range(num_atoms):
                parts = lines[i].split()
                x, y, z = map(float, parts[1:4])
                fx, fy, fz = map(float, parts[7:10])
                config.append([x, y, z])
                forces.append([fx, fy, fz])
                i += 1
            data.append(config)
            data_forces.append(forces)
        return torch.tensor(data, dtype=dtype), torch.tensor(data_forces,dtype=dtype)

class PropaneDataset_mace_deriv(Dataset):
    def __init__(self,position_path, energy_path, mean_std=None, datatype=torch.float32):
        self.datatype = datatype
        # load position data
        self.pos_data, self.forces = load_xyz_energy_force(position_path)
        # load energies -> target
        self.energy_data = torch.tensor(np.loadtxt(energy_path, skiprows=1)[:,2], dtype=datatype)
        # create SOAP descriptors
        self.soap = SOAP(
        species=["C", "H"],
        r_cut=4.0,
        n_max=8,
        l_max=8,
        sigma=0.125,
        periodic=False,
        )
        if mean_std is not None:
            mean, std = mean_std
            self.energy_data = (self.energy_data - mean) / std  

    def __len__(self): 
        return len(self.energy_data)
    
    def __getitem__(self, idx):
        propane = Atoms('C3H8', positions=self.pos_data[idx])
        derivatives, descriptors = self.soap.derivatives(propane)
        der = torch.tensor(derivatives, dtype=self.datatype)
        desc = torch.tensor(descriptors, dtype=self.datatype)
        return desc, self.energy_data[idx], self.pos_data[idx], der, self.forces[idx]


class PropaneDataset_CG(Dataset):
    def __init__(self,position_path, energy_path, model, device):
        # load position data
        self.position_data, _ = load_xyz_energy_force(position_path)
        # load energies -> target
        self.energy_data = torch.tensor(np.loadtxt(energy_path, skiprows=1)[:,2], dtype=torch.float32)
        # create SOAP descriptors
        soap = SOAP(
        species=["C", "H"],
        r_cut=4.0,
        n_max=8,
        l_max=8,
        sigma=0.125,
        periodic=False,
        )
        # calculate descriptors
        desc = []
        self.rep = []
        for i in range(len(self.energy_data)):
            propane = Atoms('C3H8', positions=self.position_data[i])
            desc.append(soap.create(propane))
        desc_array = np.array(desc)
        desc = torch.tensor(desc_array, dtype=torch.float32).to(device)
        self.energy_data = self.energy_data
        batch_size = 100
        num_batches = int(np.ceil(len(self.energy_data) / batch_size))

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.energy_data))
            
            batch_desc = desc[start_idx:end_idx, ...]
            batch_positions = torch.tensor(self.position_data[start_idx:end_idx, ...], dtype=torch.float32).to(device)
            
            energy, encoded = model.teacher_step([batch_desc, batch_positions])
            self.rep.append(encoded.detach())
        self.rep = torch.cat(self.rep, dim=0)

    def __len__(self): 
        return len(self.energy_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.position_data[idx], dtype=torch.float32), self.rep[idx], self.energy_data[idx], 