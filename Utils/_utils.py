import os, time, logging
import tqdm, yaml
from ..FileSystem import ensuredir
import typing

__all__ = [
    'listt', 'time_stamp', 'print_command', 'powerline_style', 'Logger', 'cover_print'
]

listt = lambda x: list(map(list, zip(*x)))
time_stamp = lambda : time.strftime('%Y%m%d-%H%M%S')

def cover_print(s, end=False):
    print(s.ljust(os.get_terminal_size().columns), end=('\n' if end else '\r'))


def print_command(title, command):
    aligen_len = len(command)
    single_side = aligen_len - len(title)
    single_side //= 2
    single_side -= 1
    print('-'*single_side + ' ' + title + ' ' + '-'*single_side)
    print(command)
    print('-'*aligen_len)

def evaluate(func):
    return func()




def powerline_style(fore, back, strings):
    assert len(fore) == len(back) and len(back) == len(strings)
    PNT = '\ue0b0'
    FMT = '\033[{}m'
    ret = ''
    ret += FMT.format('{};{}'.format(fore[0] + 30, back[0] + 40))
    ret += strings[0]
    for i in range(1, len(fore)):
        ret += FMT.format('{};{}'.format(back[i-1] + 30, back[i] + 40)) + PNT
        ret += FMT.format(fore[i] + 30)
        ret += strings[i]
    ret += '\033[0;{}m'.format(back[-1] + 30) + PNT + '\033[0m'
    return ret


class Logger:
    '''自定义Logger：
    使用方法：指定 Logger 名称和存储目录，建立 Logger实例
    调用实例时直接对属性赋值，传入Log列表，格式同Matlab。
    如：
    lgr = Logger('winlaic')
    lgr.i = "accuracy", 0.6, "loss", 3.7
    '''
    class DIYStyle(logging.PercentStyle):
        def __init__(self, fmt_or_style):
            if fmt_or_style is None:
                self._fmt = self.default_format
            elif isinstance(fmt_or_style, str):
                self._fmt = fmt_or_style
            elif isinstance(fmt_or_style, logging.PercentStyle):
                self._fmt = fmt_or_style._fmt
            else:
                raise TypeError
        def format(self, record):
            return super().format(record)
    class PowerlineStyle(DIYStyle):
        TIME = [0, 4]
        LEVEL = {
            'DEBUG':    [0, 6],
            'INFO':     [0, 2],
            'WARNING':  [0, 3],
            'ERROR':    [7, 1]
        }
        
        def format(self, record):
            return powerline_style(
                [self.TIME[0], self.LEVEL[record.levelname][0]],
                [self.TIME[1], self.LEVEL[record.levelname][1]],
                [' ' + record.asctime + ' ', ' ' + record.levelname + ' ']
            ) + ' ' + record.message
    class PlainStyle(DIYStyle):
        FMT_LEVEL = {
            'DEBUG': ' ',
            'INFO': ' ',
            'WARNING': '',
            'ERROR': ' '
        }
        FMT_LEVEL_CLR = {
            'DEBUG': ' ',
            'INFO': '  ',
            'WARNING': '',
            'ERROR': ' ',
        }
        def format(self, record):
            record.levelname = self.FMT_LEVEL[record.levelname] + record.levelname + self.FMT_LEVEL_CLR[record.levelname]
            return super().format(record)
    class ColoredStyle(DIYStyle):
        FMT_CLR = '\033[0m'
        FMT_TIME = '\033[1;97m'
        FMT_LEVEL = {
            'DEBUG': ' \033[1;96m',
            'INFO': ' \033[1;92m',
            'WARNING': '\033[1;93m',
            'ERROR': ' \033[1;91m'
        }
        FMT_LEVEL_CLR = {
            'DEBUG': '\033[0m ',
            'INFO': '\033[0m  ',
            'WARNING': '\033[0m',
            'ERROR': '\033[0m ',
        }
        def format(self, record):
            record.asctime = self.FMT_TIME + record.asctime + self.FMT_CLR
            record.levelname = self.FMT_LEVEL[record.levelname] + record.levelname + self.FMT_LEVEL_CLR[record.levelname]
            return super().format(record)
        
    # 与 TQDM 配合的 Handler ，防止干扰进度条的打印。            
    class TqdmLoggingHandler(logging.Handler):

        def __init__(self,level = logging.NOTSET):
            super().__init__(level)
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)
    def __init__(self, logger_name='log', loggers_directory='.', write_to_file=True):

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        basic_fmt = '%(asctime)s | %(levelname)s | %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'

        self.stream_handler = Logger.TqdmLoggingHandler()
        self.stream_handler.setLevel(logging.DEBUG)
        self.stream_formatter = logging.Formatter(basic_fmt, datefmt=datefmt)
        self.stream_formatter._style = Logger.PowerlineStyle(self.stream_formatter._style)
        self.stream_handler.formatter = self.stream_formatter
        self.logger.addHandler(self.stream_handler)

        if write_to_file:
            file_name = time_stamp() + '.log'
            file_path = ensuredir(loggers_directory, logger_name, file_name=file_name)
            
            self.file_handler = logging.FileHandler(file_path)
            self.file_handler.setLevel(logging.INFO)
            self.basic_formatter = logging.Formatter(basic_fmt, datefmt=datefmt)
            self.basic_formatter._style = Logger.PlainStyle(self.basic_formatter._style)
            self.file_handler.formatter = self.basic_formatter
            self.logger.addHandler(self.file_handler)

    def parse_message(self, message):
        if isinstance(message, typing.Mapping):
            ret = '\t'.join(['{}={}'.format(k, v) for k, v in message.items()])
        elif isinstance(message, (typing.List, typing.Tuple)):
            ret = '\t'.join([str(item) for item in message])
        else:
            ret = str(message)
        return ret
        

    @property
    def d(self): pass

    @property
    def i(self): pass

    @property
    def w(self): pass

    @property
    def e(self): pass

    @w.setter
    def w(self, message_list):
        self.logger.warning(self.parse_message(message_list))

    @i.setter
    def i(self, message_list):
        self.logger.info(self.parse_message(message_list))

    @d.setter
    def d(self, message_list):
        self.logger.debug(self.parse_message(message_list))

    @e.setter
    def e(self, message_list):
        self.logger.error(self.parse_message(message_list))
