'''
共用的函数
'''

def db_to_amplitude(db_val):
    """
    将分贝转换为线性幅度值。
    公式：A = 10^(dB/20)
    """
    return 10 ** (db_val / 20.0)
