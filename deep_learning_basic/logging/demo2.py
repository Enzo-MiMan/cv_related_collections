import logging


# ============================ 1、实例化 logger ============================
# 实例化一个记录器，使用默认记录器名称 'root'，并将日志级别设为 info
logger = logging.getLogger("training.loss.log")
logger.setLevel(logging.INFO)


# ============================ 2、定义Handler ============================
# 创建一个往 console打印输出的 Handler，日志级别设为 debug
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 再创建一个往文件中打印输出的handler， 默认使用记录器同样的日志级别
file_handler = logging.FileHandler(filename='demo.log', mode='w')


# ============================ 3、定义打印格式 ============================
# 创建一个标准版日志打印格式
standard_formatter = logging.Formatter('%(asctime)s %(name)s [%(pathname)s line:(lineno)d] %(levelname)s %(message)s')

# 创建一个简单版日志打印格式
simple_formatter = logging.Formatter('%(levelname)s %(message)s')

# ============================ 4、定义打过滤器 ============================
# 实例化一个过滤器
flt = logging.Filter('training.accurate')

# ============================ 5、绑定关系 ============================
# 让 consoleHandler 使用 标准版日志打印格式
console_handler.setFormatter(standard_formatter)

# 让 fileHandler 使用 简版日志打印格式
file_handler.setFormatter(simple_formatter)

# 给记录器绑定上 consoleHandler 和 fileHandler
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 将过滤器flt绑定到 console_handler
console_handler.addFilter(flt)

# ============================ 5、打印 ============================
logger.debug('调试日志')
logger.info('消息日志')
logger.warning('警告日志')
logger.error('错误日志 ')
logger.critical('严重错误日志')


