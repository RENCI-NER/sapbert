from bs4 import BeautifulSoup


class MESH:
    def __init__(self, mesh_xml_file_with_path):
        self.mesh_file = mesh_xml_file_with_path
        self.names = []
        self.ids = []

    def load_mesh(self):
        with open(self.mesh_file, 'r') as f:
            data = f.read()

        bs_data = BeautifulSoup(data, 'xml')
        names = bs_data.find_all('DescriptorName')
        ids = bs_data.find_all('DescriptorUI')
        for name_xml, id_xml in zip(names, ids):
            self.names.append(name_xml.get_text())
            self.ids.append(id_xml.get_text())
