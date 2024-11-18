PROMPTS_NOMINALIZATION = {
    "target": [
        '''Nominalized adjective:''',
        '''Noun:''',
        '''The following is a nominalized adjective:''',
        '''The following is a noun:'''
    ],
    "base": [
        '''{} ->''',
        '''{} :''',
        '''{} -''',
        '''{}'''
    ],
    "instruction": [
        '''Adjective: {}\nNominalization:''',
        '''Form the nominalization of the given adjective.\n\n{} ->''',
        '''Nominalize the given adjective.\n\n{} ->''',
        '''Turn the given adjective into a noun.\n\n{} ->'''
    ]
}


AFFIXES2PROMPTS = {
    "ity_ness_nonce": PROMPTS_NOMINALIZATION,
    "ity_ness_seen": PROMPTS_NOMINALIZATION
}
