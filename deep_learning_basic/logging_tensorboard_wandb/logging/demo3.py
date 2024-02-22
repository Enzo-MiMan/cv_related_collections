import logging.config

logging.config.fileConfig('demo.conf')

root_logger = logging.getLogger()
my_logger = logging.getLogger('mylogger')

root_logger.debug('调试日志')
root_logger.info('消息日志')
root_logger.warning('警告日志')
root_logger.error('错误日志 ')
root_logger.critical('严重错误日志')

my_logger.debug('调试日志')
my_logger.info('消息日志')
my_logger.warning('警告日志')
my_logger.error('错误日志 ')
my_logger.critical('严重错误日志')