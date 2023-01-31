import sys
from tqdm import tqdm
from glob import glob
from scapy.all import PcapReader, IP
import numpy as np

if __name__ == "__main__":
    dataset_path = sys.argv[1]

    malware_path = dataset_path + "Malware/"
    benign_path = dataset_path + "Benign/"


    malware_ttls = []
    for malware_pcap in tqdm(glob(malware_path + '*.pcap')):
        for packet in PcapReader(malware_pcap):
            if packet.haslayer(IP):
                malware_ttls.append(packet[IP].ttl)
    
    malware_mean = np.mean(malware_ttls)
    malware_stddev = np.std(malware_ttls)

    benign_ttls = []
    for benign_pcap in tqdm(glob(benign_path + '*.pcap')):
        for packet in PcapReader(benign_pcap):
            if packet.haslayer(IP):
                benign_ttls.append(packet[IP].ttl)
    
    benign_mean = np.mean(benign_ttls)
    benign_stddev = np.std(benign_ttls)

    print(f"Malware mean ttl: {malware_mean}")
    print(f"Malware ttl standard deviation: {malware_stddev}")
    print(f"Benign mean ttl: {benign_mean}")
    print(f"Benign ttl standard deviation: {benign_stddev}")
    


            
