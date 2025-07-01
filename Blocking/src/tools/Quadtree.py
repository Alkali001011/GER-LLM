import math

from Blocking.src.tools.utils import load_pkl


def reassign_value2static_variable(aoi2poi, poi2aoi, poi_id2se_address):
    Quadtree.aoi2poi = aoi2poi
    Quadtree.poi2aoi = poi2aoi
    Quadtree.poi_id2se = load_pkl(poi_id2se_address)


def reassign_value2static_variable_aqsb(aoi2poi_address, poi2aoi_address, poi_id2se_address):
    Quadtree.aoi2poi = load_pkl(aoi2poi_address)
    Quadtree.poi2aoi = load_pkl(poi2aoi_address)
    Quadtree.poi_id2se = load_pkl(poi_id2se_address)


class Quadtree:
    aoi2poi = None
    poi2aoi = None
    poi_id2se = None

    def __init__(self, level, x, y, width, height):
        self.internal_se = set()  # 经纬度坐标严格属于该Quadtree内部的实体集合
        self.external_se = set()  # 经纬度坐标严格属于该Quadtree外部的实体集合
        self.internal_aoi_id = set()  # 该四叉树内的实体所属的aoi

        self.X = x  # 区域左上角x坐标值
        self.Y = y  # 区域左上角y坐标值
        self.Width = width  # 区域宽度
        self.Height = height  # 区域高度

        self.nodes = [None] * 4  # 当前四叉树节点的子节点,初始化为None
        self.level = level  # 当前四叉树所在的层级

    def insert(self, se):
        if (self.X <= se.x <= self.X + self.Width) and (self.Y <= se.y <= self.Y + self.Height):
            self.internal_se.add(se)

    def cal_diagonal(self):
        return math.sqrt(self.Width * self.Width + self.Height * self.Height)

    def cal_density(self):
        # merge_set = self.internal_se | self.external_se
        # return len(merge_set)/(self.Width * self.Height)
        return len(self.internal_se) / (self.Width * self.Height)

    def cal_number_of_aoi(self):
        return len(self.internal_aoi_id)

    def internal_get_index(self, se):
        """
        :param se: 经纬度坐标严格属于该Quadtree内部的实体
        :return: 按经纬度，它应该属于哪个子块
        """
        vertical_half = self.X + self.Width / 2
        horizontal_half = self.Y + self.Height / 2

        # if se.x > vertical_half and se.y < horizontal_half:
        #     return 0
        # if se.x <= vertical_half and se.y <= horizontal_half:
        #     return 1
        # if se.x < vertical_half and se.y > horizontal_half:
        #     return 2
        # if se.x >= vertical_half and se.y >= horizontal_half:
        #     return 3

        if vertical_half < se.x <= self.X + self.Width and self.Y <= se.y <= horizontal_half:
            return 0
        if self.X <= se.x <= vertical_half and self.Y <= se.y < horizontal_half:
            return 1
        if self.X <= se.x < vertical_half and horizontal_half <= se.y <= self.Y + self.Height:
            return 2
        if vertical_half <= se.x <= self.X + self.Width and horizontal_half < se.y <= self.Y + self.Height:
            return 3

    def split(self):
        # 1.切分成四块,初始化子节点
        sub_width = self.Width / 2
        sub_height = self.Height / 2
        x = self.X
        y = self.Y

        self.nodes[0] = Quadtree(self.level + 1, x + sub_width, y, sub_width, sub_height)
        self.nodes[1] = Quadtree(self.level + 1, x, y, sub_width, sub_height)
        self.nodes[2] = Quadtree(self.level + 1, x, y + sub_height, sub_width, sub_height)
        self.nodes[3] = Quadtree(self.level + 1, x + sub_width, y + sub_height, sub_width, sub_height)

        # 2.将内部实体放到该放的子节点去
        for se in self.internal_se:
            index = self.internal_get_index(se)
            self.nodes[index].internal_se.add(se)
            if not Quadtree.poi2aoi.__contains__(se.id):
                continue
            self.nodes[index].internal_aoi_id.update(Quadtree.poi2aoi[se.id])  # 同时更新内部aoi列表

        self.internal_se.clear()

        # 3.处理外部结点归属
        if len(self.external_se) != 0:  # 如果存在外部节点
            for se in self.external_se:  # 对每一个外部节点
                if not Quadtree.poi2aoi.__contains__(se.id):
                    continue
                aoi_id_list = Quadtree.poi2aoi[se.id]  # 获得它所处的aoi的list
                for aoi_id in aoi_id_list:  # 对每一个aoi id
                    for i in range(0, 4):  # 遍历所有子块
                        if aoi_id in self.nodes[i].internal_aoi_id:
                            self.nodes[i].external_se.add(se)
            self.external_se.clear()

        # 4.给每个子块的内部实体作aoi闭包，新增的实体放到对应子块的外部实体集合
        for i in range(0, 4):  # 遍历子块
            for aoi_id in self.nodes[i].internal_aoi_id:  # 遍历子块内的aoi id
                se_id_list = Quadtree.aoi2poi[aoi_id]  # 获得其下属poi id的list
                set_se = set()  # 将实体id的列表转变为实体对象的列表
                for se_id in se_id_list:
                    set_se.add(Quadtree.poi_id2se[se_id])
                self.nodes[i].external_se.update(set_se - self.nodes[i].internal_se)  # 这里为了防止把内部实体也加到外部，需要做一个减法
