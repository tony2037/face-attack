from loader import Loader

loader = Loader('faces94/malestaff/voudcx', 'faces94/malestaff/tony')
a, b = loader.get_all()
print(a.shape)
print(b.shape)
