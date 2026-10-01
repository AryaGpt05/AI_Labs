% CS-F407: Artificial Intelligence
% Laboratory - Logical Reasoning for Planning
% Prolog Verification Knowledge Base
% Author: Arya Gupta

% ----------------------------------------------------------------------
% Warehouse Connectivity Facts
% ----------------------------------------------------------------------
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% ----------------------------------------------------------------------
% Movement & Verification Rules
% ----------------------------------------------------------------------
can_move(X, Y) :-
    connected(X, Y).

valid_move(X, Y) :-
    connected(X, Y).

% ----------------------------------------------------------------------
% Task 8: Wet Road Logical Inference Example
% ----------------------------------------------------------------------
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.

% Example Verification Queries:
% ?- can_move(a, b).    % Expected: true.
% ?- can_move(a, c).    % Expected: false.
% ?- valid_move(a, b).  % Expected: true.
% ?- valid_move(b, c).  % Expected: true.
% ?- valid_move(a, c).  % Expected: false.
% ?- reduce_speed.      % Expected: true.
