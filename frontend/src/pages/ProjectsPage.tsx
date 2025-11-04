import ButtonLink from '../components/ButtonLink';
import PageSection from '../components/PageSection';

const projects = [
  {
    title: 'AtlasSight',
    caption: 'Autonomous inspection',
    description:
      'Vision-first pipeline for industrial inspection. Includes automated defect tagging, active learning loops and gRPC inference microservices.'
  },
  {
    title: 'PulseStudio',
    caption: 'Medical imaging acceleration',
    description:
      'High-throughput segmentation service tuned for MRI and CT. TensorRT inference, dynamic batching and observability dashboards accessible to clinicians.'
  },
  {
    title: 'VistaFlow',
    caption: 'Retail analytics',
    description:
      'Edge-ready detection models with on-device optimization. Powered by quantization aware training and offline evaluation harnesses built within K3Project.'
  }
];

const ProjectsPage = () => {
  return (
    <div className="page-stack">
      <PageSection
        subtitle="Portfolio"
        title="Selected launches by Nik52"
        description="Each project stems from experiments inside the K3Project lab and scales with a focus on clarity and maintainability."
      >
        <div className="card-grid">
          {projects.map((project) => (
            <article className="card" key={project.title}>
              <span className="badge">{project.caption}</span>
              <h3 className="card__title">{project.title}</h3>
              <p className="card__description">{project.description}</p>
            </article>
          ))}
        </div>
      </PageSection>

      <PageSection
        subtitle="More"
        title="Let us build your next CV product"
        description="I collaborate with teams who value focused iteration, actionable metrics and design-first delivery."
        align="center"
      >
        <div className="hero__actions" style={{ justifyContent: 'center' }}>
          <ButtonLink to="/contact">Start a brief</ButtonLink>
          <ButtonLink to="/about" variant="ghost">
            Learn about Nik52
          </ButtonLink>
        </div>
      </PageSection>
    </div>
  );
};

export default ProjectsPage;
