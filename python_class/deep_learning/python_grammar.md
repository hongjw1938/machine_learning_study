## Python 문법 정리 추가
>###    Built-in function : format(value[, format_spec])
        - value를 formatted representation에 맞추어 변경함.
        - formatted은 format_spec에 의해 사용자가 controll 가능
        - default format_spec은 empty string이다. str(value)와 같은 효과를 가진다.
        - format(value, format_spec)은 type(value).__format__(value, format_spec)으로 치환된다.
        - 만약 method가 object(최상위)에 도달했는데 format_spec이 empty가 아니면 TypeError가 발생한다.
        * 포맷팅 방법
            format_spec ::= [[fill]align][sign][#][0][width][,][.precision][type]
            fill ::= <a character other than '}'>
            align ::= "<" | ">" | "=" | "^:
            sign ::= "+" | "-" | " "
            width ::= integer
            precision ::= integer
            type ::= "b" | "c" | "d" | "e" | "E" | "f" | "F" | "g" | "G" | "n" | "o" | "s" | "x" | "X" | "%"
        * 예시
            >>> format('ABCDEFG', '<30')
            'ABCDEFG                       '
            
            >>> format('ABCDEFG', '>30')
            '                       ABCDEFG'
            
            >>> format('ABCDEFG', '^30')
            '           ABCDEFG            '
            
            >>> format(1234567890, '+')
            '+1234567890'
            
            >>> format(1234567890, '-')
            '1234567890'
            
            >>> format(1234567890, ' ')
            ' 1234567890'
            
            >>> format(1234567890, ',')
            '1,234,567,890'
            
            >>> format(0xABDC1234)
            '2883326516'
            
            >>> format(0xABDC1234, 'b')
            '10101011110111000001001000110100'
            
            >>> format(0xABDC1234, 'x')
            'abdc1234'
            
            >>> format(0xABDC1234, 'X')
            'ABDC1234'
            
            >>> format(0xABDC1234, 'n')
            '2883326516'
            
            >>> format(0xABDC1234, 'o')
            