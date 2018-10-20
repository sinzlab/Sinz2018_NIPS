import textwrap


class Messager:
    _last_source = ''
    @classmethod
    def msg(cls, *msg, depth=0, **kwargs):
        source = cls.__name__
        if Messager._last_source != source:
            tmp = 27*'-' + '+' + 80*'-' + '\n'
            Messager._last_source = source
        else:
            tmp = ''

        lines = []
        for line in ' '.join(map(str, msg)).split('\n'):
            lines.extend(textwrap.wrap(line, width=80, subsequent_indent=5*" "))

        if len(source) > 25:
            source = source[:22] + '...'
        tmp += source.ljust(25)
        msg = tmp + ('  | ' + depth * '\t') \
                    + ('\n'.ljust(25) + '   | ' + depth * '\t').join(lines)
        print(msg, **kwargs)


