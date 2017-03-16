# author: Juan Zamora

import sys
import optparse
import docx2txt

parser = optparse.OptionParser()
parser.add_option('-i', '--input', dest='input', help='Input docx file')
parser.add_option('-o', '--output', dest='output', help='Output txt file')

(options, args) = parser.parse_args()

if options.input is None:
    print("Usage:"+sys.argv[0]+" -i inputfile_path [-o outputfilepath]")
    sys.exit(-1)

if options.output is None:
    options.output = options.input[:-4] + "txt"

output = docx2txt.process(options.input)
f = open(options.output, "w")
f.write(output.encode("utf8"))
f.close()