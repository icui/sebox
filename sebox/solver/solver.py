from nnodes.root import parse_import


def mesh(node):
    parse_import((f'sebox.solver.{node.solver}', 'mesh'))(node)


def forward(node):
    parse_import((f'sebox.solver.{node.solver}', 'forward'))(node)


def adjoint(node):
    parse_import((f'sebox.solver.{node.solver}', 'adjoint'))(node)


def smooth(node):
    parse_import((f'sebox.solver.{node.solver}', 'smooth'))(node)
