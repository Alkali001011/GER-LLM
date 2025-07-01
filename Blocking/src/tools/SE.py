import math


# web_mercator投影
def web_mercator_x(longitude):
    return longitude * 20037508.342789 / 180


def web_mercator_y(latitude):
    return (math.log(math.tan((90 + latitude) * math.pi / 360)) / (math.pi / 180)) * 20037508.34789 / 180 * (-1)


class SE:
    def __init__(self, id, name, address, category, longitude, latitude, belong):
        self.id = id
        self.name = name
        self.address = address
        self.category = category
        self.longitude = longitude
        self.latitude = latitude
        # 转换为Web墨卡托坐标
        self.x = web_mercator_x(self.longitude)
        self.y = web_mercator_y(self.latitude)
        # 标记实体属于哪个数据集
        self.belong = belong

    def display_properties(self):
        print(f"ID: {self.id}")
        print(f"Name: {self.name}")
        print(f"Address: {self.address}")
        print(f"Category: {self.category}")
        print(f"Longitude: {self.longitude}, Latitude: {self.latitude}")
        # print(f"X: {self.x}, Y: {self.y}")
        print(f"Belong: {self.belong}")
        print()

    def to_str(self, city):
        if city == 'pit':  # pit数据集不显示类别
            return 'Name: ' + str(self.name) + ', Address: ' + str(self.address)
        else:
            return 'Name: ' + self.name + ', Category: ' + self.category + ', Address: ' + self.address

    def __eq__(self, other):
        # 当两个实例的 id 和 belong 属性都相等时，才认为它们相等
        if not isinstance(other, SE):
            return NotImplemented
        return self.id == other.id and self.belong == other.belong

    def __hash__(self):
        # 基于 id 和 belong 属性的组合哈希值
        return hash((self.id, self.belong))
