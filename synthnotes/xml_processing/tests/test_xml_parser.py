import pytest
from ctakes_xml import contains, CtakesXmlOutput, CtakesXmlParser


class TestParser(object):

    @pytest.fixture
    def xml_out(self):
        out = CtakesXmlOutput('tests/xml_doc.xmi')
        return out

    def test_contains(self):
        e1 = {'begin': 15, 'end': 47}
        e2 = {'begin': 16, 'end': 30}
        assert contains(e1, e2) == True

    def test_doesnt_contain(self):
        e1 = {'begin': 15, 'end': 47}
        e2 = {'begin': 10, 'end': 13}
        assert contains(e1, e2) == False

    def test_contains_on_start_of_span(self):
        e1 = {'begin': 15, 'end': 47}
        e2 = {'begin': 15, 'end': 22}
        assert contains(e1, e2) == True

    def test_contains_on_end_of_span(self):
        e1 = {'begin': 15, 'end': 47}
        e2 = {'begin': 40, 'end': 47}
        assert contains(e1, e2) == True

    def test_elem_start_before_span(self):
        e1 = {'begin': 15, 'end': 47}
        e2 = {'begin': 10, 'end': 16}
        assert contains(e1, e2) == False

    def test_elem_ends_after_span(self):
        e1 = {'begin': 15, 'end': 47}
        e2 = {'begin': 35, 'end': 48}
        assert contains(e1, e2) == False

    def test_gets_doc_id(self, xml_out):
        assert xml_out.doc_id == 321511

    def test_gets_sentences(self, xml_out):
        assert len(list(xml_out.sentences)) == 10

    def test_gets_tokens(self, xml_out):
        assert len(list(xml_out.tokens)) == 139






