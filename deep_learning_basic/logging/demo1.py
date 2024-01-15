import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)s [%(pathname)s line:%(lineno)d] %(levelname)s %(message)s]',
                    # filename='demo.log',
                    # filemode='w'
                    )

logging.debug('调试日志')
logging.info('消息日志')
logging.warning('警告日志')
logging.error('错误日志 ')
logging.critical('严重错误日志')
