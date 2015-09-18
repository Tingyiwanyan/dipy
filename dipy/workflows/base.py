from dipy.fixes import argparse as arg
import inspect


try:
    import numpydoc as ndoc
    has_ndoc = True
except ImportError:
    print('Numpydoc is not installed. \n pip install numpydoc')
    has_ndoc = False


class IntrospectiveArgumentParser(arg.ArgumentParser):

    def __init__(self, prog=None, usage=None, description=None, epilog=None,
                 version=None, parents=[],
                 formatter_class=arg.RawTextHelpFormatter,
                 prefix_chars='-', fromfile_prefix_chars=None,
                 argument_default=None, conflict_handler='resolve',
                 add_help=True):

        """ Augmenting the argument parser to allow automatic creation of
        arguments from workflows

        Parameters
        -----------
        prog : None
            The name of the program (default: sys.argv[0])
        usage : None
            A usage message (default: auto-generated from arguments)
        description : str
            A description of what the program does
        epilog : str
            Text following the argument descriptions
        version : None
            Add a -v/--version option with the given version string
        parents : list
            Parsers whose arguments should be copied into this one
        formatter_class : obj
            HelpFormatter class for printing help messages
        prefix_chars : str
            Characters that prefix optional arguments
        fromfile_prefix_chars : None
            Characters that prefix files containing additional arguments
        argument_default : None
            The default value for all arguments
        conflict_handler : str
            String indicating how to handle conflicts
        add_help : bool
            Add a -h/-help option
        """

        iap = IntrospectiveArgumentParser
        super(iap, self).__init__(prog, usage, description, epilog, version,
                                  parents, formatter_class, prefix_chars,
                                  fromfile_prefix_chars, argument_default,
                                  conflict_handler, add_help)

        self.doc = None

    def add_workflow(self, workflow):
        specs = inspect.getargspec(workflow)
        doc = inspect.getdoc(workflow)

        if has_ndoc:
            self.doc = ndoc.docscrape.NumpyDocString(doc)['Parameters']

        args = specs.args
        defaults = specs.defaults

        len_args = len(args)
        len_defaults = len(defaults)

        # Arguments with no defaults (Positional)
        cnt = 0
        for i in range(len_args - len_defaults):
            if has_ndoc:
                help_msg = ''.join(self.doc[i][2])
                self.add_argument(args[i], help=help_msg)
            else:
                self.add_argument(args[i])
            cnt += 1

        # Arguments with defaults (Optional)
        for i in range(cnt, len_args):
            if has_ndoc:
                typestr = self.doc[i][1]
                dtype = self._select_dtype(typestr)
                help_msg = ' '.join(self.doc[i][2])
                self.add_argument('--' + args[i], action='store', type=dtype,
                                  metavar=dtype.__name__, help=help_msg)
            else:
                self.add_argument('--' + args[i])

    def _select_dtype(self, text):
        text = text.lower()
        if text.find('str') != -1:
            return str
        if text.find('int') != -1:
            return int
        if text.find('float') != -1:
            return float
        if text.find('bool') != -1:
            return bool

    def get_flow_args(self, args=None, namespace=None):
        ns_args = self.parse_args(args, namespace)
        dct = vars(ns_args)

        return dict((k, v) for k, v in dct.items() if v is not None)

    def update_argument(self, *args, **kargs):

        self.add_argument(*args, **kargs)

    def show_argument(self, dest):

        for act in self._actions[1:]:
            if act.dest == dest:
                print(act)

    def add_epilogue(self):
        # with citations
        pass

    def add_description(self):
        pass


